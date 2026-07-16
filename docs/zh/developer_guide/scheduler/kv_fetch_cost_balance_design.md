# 全局 KV 池化场景下的读取成本与计算均衡调度设计

## 1. 文档概述

本文设计一种适用于全局 KV 池化场景的请求调度策略
`KVFetchCostBalancePolicy`。该策略不再使用不同 Data Parallel（DP）
间的 KV 命中率作为缓存亲和性指标，而是同时预测请求在各 DP 上的
KV 读取时间和计算完成时间，以预计请求完成时间最小化为调度目标。

### 1.1 背景

全局 KV 池化后，同一请求分发到不同 DP 时通常能够访问相同的 KV
数据，因此各 DP 的命中 token 数和命中率可能完全一致。在这种情况下：

- KV 命中率不能区分候选 DP；
- 数据来源、网络拓扑和资源竞争仍可能导致不同的 KV 读取时间；
- 只按 KV 命中率调度会退化为随机选择，甚至因预测噪声制造热点；
- 只按当前请求数或 token 数调度，又可能忽略远端 KV 读取造成的
  TTFT 和共享网络压力。

因此，本设计将传统的“缓存亲和性”重新定义为“KV 获取成本感知”，并将
KV 获取成本与计算负载统一转换为时间量纲。

### 1.2 设计目标

- 在 KV 命中率相同的候选 DP 之间识别实际读取成本差异；
- 同时控制 DP 计算热点和 KV 存储、网络热点；
- 优先优化 TTFT，并提高满足 SLO 的集群吞吐；
- 在预测缺失、状态陈旧或模型失准时安全降级；
- 避免固定比例组合命中率和负载分数所需的反复调参；
- 支持 PD 分离和 Prefill/Decode 共置部署。

### 1.3 非目标

- 不负责 KV 对象的存储、复制和淘汰策略；
- 不改变推理引擎内部的 batch 调度算法；
- 不保证一次预测能够精确还原所有硬件流水线时序；
- 第一阶段不引入复杂的深度学习预测模型。

## 2. 核心判断

### 2.1 命中率相同不代表读取成本相同

设请求 \(r\) 在全局 KV 池中的命中 token 数为 \(H_r\)。对任意候选
DP \(d\)，有：

\[
H_{r,d} = H_r
\]

但不同 DP 的读取时间仍可能不同：

\[
R_{r,d_1} \ne R_{r,d_2}
\]

差异来自 KV 所在节点、网络跳数、NIC/PCIe 竞争、源节点负载和传输
排队等因素。因此，调度器需要优化的是 \(R_{r,d}\)，而不是
\(H_{r,d}\)。

### 2.2 读取成本相同时应退化为计算均衡

如果所有候选 DP 的预测读取时间差小于预测噪声：

\[
\max_d \hat R_{r,d} - \min_d \hat R_{r,d} < \epsilon_R
\]

调度器应忽略 KV 读取项，按预计计算完成时间分发请求。此机制可避免
读取时间模型的微小误差主导调度决策。

## 3. KV 读取时间评估难点

| 难点 | 影响 | 解决方向 |
|------|------|----------|
| KV 来源层级动态变化 | HBM、主机内存、远端内存和持久化存储延迟差异大 | 在 KV 查询结果中携带来源层级和源节点 |
| 有效带宽动态变化 | 峰值带宽无法表示并发竞争后的实际吞吐 | 使用短窗口实际吞吐和并发流量预测有效带宽 |
| 传输前排队 | 存储服务、RDMA 队列和源节点可能成为瓶颈 | 分别记录 enqueue、start 和 end 时间 |
| 拓扑非对称 | 不同 DP 到 KV 源节点的链路和 NUMA 路径不同 | 维护源节点到目标 DP 的拓扑特征 |
| KV 字节数估算误差 | TP 切分、压缩和块对齐导致 token 到字节换算不稳定 | 优先使用传输层返回的计划读取字节数 |
| 读取与计算重叠 | 简单相加会高估服务时间 | 估计重叠比例或直接学习非重叠读取时间 |
| 尾延迟显著 | 均值预测无法保护 TTFT SLO | 同时预测 P50、P90 和 P99 |
| 状态观测滞后 | 多请求可能同时选择同一 DP 或 NIC | 服务端权威仲裁并预留虚拟资源 |
| 冷启动样本不足 | 新 DP、新拓扑没有历史数据 | 使用解析模型和保守带宽初始化 |
| 异常值污染 | 重试、故障切换可能产生极端样本 | 使用分位数、截尾和异常标签隔离 |

## 4. 数据采集与可观测性

### 4.1 单次 KV 读取事件

每次读取应至少记录以下字段：

| 字段 | 说明 |
|------|------|
| `request_id` | 请求标识 |
| `source_tier` | KV 来源层级 |
| `source_node` | KV 源节点 |
| `target_instance_id` | 目标实例 |
| `target_dp_rank` | 目标 DP |
| `matched_tokens` | 命中 token 数 |
| `planned_bytes` | 计划读取字节数 |
| `actual_bytes` | 实际读取字节数 |
| `enqueue_timestamp` | 进入读取队列时间 |
| `start_timestamp` | 实际开始读取时间 |
| `first_byte_timestamp` | 首字节到达时间 |
| `end_timestamp` | 读取完成时间 |
| `compute_start_timestamp` | 计算开始时间 |
| `compute_end_timestamp` | 计算结束时间 |
| `concurrent_reads` | 同期读取任务数 |
| `source_pending_bytes` | 源节点待读取字节数 |
| `target_pending_bytes` | 目标节点待读取字节数 |
| `retry_count` | 传输重试次数 |
| `error_type` | 错误或降级类型 |

由上述时间戳派生：

\[
T_{queue} = T_{start} - T_{enqueue}
\]

\[
T_{transfer} = T_{end} - T_{start}
\]

\[
BW_{effective} = \frac{actual\_bytes}{T_{transfer}}
\]

### 4.2 DP 计算状态

调度器需要维护：

- 等待和运行中的 prefill token 数；
- 当前 batch size 和 batch 中的序列长度分布；
- 最近窗口的 prompt TPS；
- GPU/NPU 利用率和计算单元繁忙度；
- Decode batch 对共置 Prefill 的干扰；
- 已分配但尚未进入引擎的预留计算量；
- 每个 DP 的虚拟完成时间。

### 4.3 网络与 KV 源状态

- 源节点和目标节点 NIC 利用率；
- 源节点正在服务的读取任务数；
- 源节点、目标节点的待读取字节；
- RDMA 重传、超时和拥塞指标；
- 源节点到目标 DP 的近期 P50/P90 吞吐；
- KV 存储层的队列长度和淘汰状态。

## 5. KV 读取时间模型

### 5.1 解析基础模型

请求 \(r\) 发往 DP \(d\) 的基础读取时间为：

\[
\hat R^{base}_{r,d}
=
\hat T^{queue}_{r,d}
+
T^{setup}_{tier,topology}
+
\frac{Bytes_r}{\hat{BW}^{effective}_{r,d}}
+
\hat T^{copy}_{r,d}
\]

其中：

\[
Bytes_r = H_r \times BytesPerToken_{KV}
\]

如果传输层能够提供 `planned_bytes`，应直接使用该值，避免根据模型结构
重复估算。

### 5.2 在线残差修正

解析模型无法覆盖所有竞争和软件栈开销，因此增加在线残差：

\[
\hat R_{r,d}
=
\hat R^{base}_{r,d}
+
\hat R^{residual}_{r,d}
\]

第一阶段推荐采用分桶 EWMA：

- 按来源层级、源/目标拓扑、KV 大小区间和并发读取区间分桶；
- 每个桶维护读取时间和有效吞吐的 P50/P90；
- 新样本使用指数衰减更新；
- 样本不足时回退到更粗粒度的父桶；
- 无历史样本时使用保守链路带宽。

后续可使用在线分位数回归或 GBDT 替代 EWMA，但必须保留解析模型作为
冷启动和降级路径。

### 5.3 风险修正

调度使用风险读取时间：

\[
R^{risk}_{r,d}
=
R^{P50}_{r,d}
+
\beta
\left(
R^{P90}_{r,d} - R^{P50}_{r,d}
\right)
\]

- 模型稳定时可取较小 \(\beta\)；
- 样本不足、网络拥塞或 SLO 紧张时增大 \(\beta\)；
- 默认建议从 \(\beta=0.5\) 开始离线标定。

## 6. 计算负载模型

### 6.1 Prefill 计算量

请求需要新计算的 token 数为：

\[
N_r = L_r - H_r
\]

其中 \(L_r\) 是完整输入 token 数。预计计算时间：

\[
C_{r,d}
=
f_d(
N_r,
L_r,
BatchState_d,
Model,
Hardware,
DecodeInterference_d
)
\]

第一阶段可以使用分段线性模型：

\[
C_{r,d}
=
a_d
+
b_d N_r
+
c_d N_r L_r
\]

参数通过各 DP 最近的真实 prefill 耗时在线校正。

### 6.2 虚拟计算队列

每个 DP 维护虚拟完成时间 \(V_d\)。分配请求前：

\[
Q_d = \max(0, V_d - now)
\]

请求分配后立即预留：

\[
V_d
\leftarrow
\max(V_d, now) + C_{r,d}
\]

实际请求完成后，根据预测误差修正 \(V_d\) 和模型参数。预留动作必须在
Scheduler Server 的权威仲裁路径中原子执行，防止多个 API Worker
同时将请求分发到同一 DP。

### 6.3 不同部署模式

#### PD 分离

- Prefill DP 主要考虑 KV 读取、等待时间和新 token 计算时间；
- Decode DP 单独使用 decode 活跃序列、预计输出长度和 batch 吞吐模型；
- 本文的 KV 获取成本主要用于 Prefill DP 选择。

#### Prefill/Decode 共置

- 需要加入当前 decode batch 对 prefill 的干扰；
- 可根据活跃 decode 序列数和近期 TPOT 估计额外计算时间；
- SLO 目标需要同时约束 TTFT 和 TPOT。

## 7. 读取与计算重叠

设读取和计算的重叠比例为 \(\rho_d \in [0,1]\)，请求在 DP \(d\)
上的预计服务时间为：

\[
T_{r,d}
=
Q_d
+
R^{risk}_{r,d}
+
C_{r,d}
-
\rho_d
\min(
R^{risk}_{r,d},
C_{r,d}
)
\]

- \(\rho_d=0\)：读取和计算完全串行；
- \(\rho_d=1\)：两者最大程度重叠；
- \(\rho_d\) 应按模型、KV 块大小、来源层级和部署模式分桶估计；
- 如果无法获得可靠的重叠埋点，第一阶段使用 \(\rho_d=0\)，保证预测
  偏保守。

## 8. 调度决策

### 8.1 候选过滤

依次过滤：

1. DP 和 endpoint 状态正常；
2. 模型、角色和租户匹配；
3. 预计计算负载不超过集群均值的配置倍数；
4. 源节点或目标 NIC 未超过拥塞阈值；
5. P90 TTFT 满足 SLO；如果全部不满足，则保留预计违约最小的候选。

### 8.2 统一时间评分

对剩余候选计算：

\[
Score_{r,d} = T_{r,d}
\]

因为 KV 读取和计算负载都转换为毫秒，无需再组合“缓存权重”和“负载
权重”。选择：

\[
d^* = \arg\min_d Score_{r,d}
\]

### 8.3 网络外部性修正

一次 KV 读取不仅影响当前请求，还会增加共享源节点和网络链路上其他
请求的等待时间。可选地加入影子价格：

\[
Score'_{r,d}
=
Score_{r,d}
+
\lambda_s \Delta PendingBytes_{source}
+
\lambda_n \Delta PendingBytes_{network}
\]

\(\lambda_s\) 和 \(\lambda_n\) 根据源节点及网络利用率动态调整，而不是
长期使用固定常数。

### 8.4 噪声抑制

- 如果最优和次优候选得分差小于 3%～5%，在近似候选中轮询；
- 只有预计收益超过 `min_switch_improvement` 时才改变已有会话的 DP；
- 对短时间内反复迁移的会话增加冷却时间；
- 使用确定性哈希或轮询打破完全相同的得分。

## 9. 调度流程

```text
请求到达
  │
  ├─ 查询全局 KV 元数据
  │    └─ 命中 token、计划读取字节、来源层级和源节点
  │
  ├─ 获取 Scheduler Server 权威快照
  │    ├─ DP 虚拟计算队列
  │    ├─ DP batch / TPS / 利用率
  │    └─ 源节点与网络待读取字节
  │
  ├─ 针对每个 DP 预测
  │    ├─ KV 读取 P50/P90
  │    ├─ 计算排队时间
  │    ├─ 请求计算时间
  │    └─ 读取计算重叠时间
  │
  ├─ 执行健康、过载、拥塞和 SLO 过滤
  │
  ├─ 读取成本差小于噪声？
  │    ├─ 是：仅按计算完成时间选择
  │    └─ 否：按统一预计完成时间选择
  │
  ├─ Scheduler Server 使用最新状态重新仲裁
  │
  ├─ 原子预留计算时间和待读取字节
  │
  └─ 请求完成后上报实际读取与计算耗时
```

### 9.1 伪代码

```python
def select_dp(request, candidates, snapshot):
    kv_plan = global_kv_pool.lookup(request)
    estimates = []

    for dp in candidates:
        read = read_predictor.predict_quantiles(
            kv_plan=kv_plan,
            target_dp=dp,
            snapshot=snapshot,
        )
        queue_ms = compute_predictor.queue_time(dp, snapshot)
        compute_ms = compute_predictor.request_time(request, kv_plan, dp, snapshot)
        overlap_ms = overlap_predictor.predict(read.p90, compute_ms, dp)

        finish_ms = (
            queue_ms
            + risk_adjust(read)
            + compute_ms
            - overlap_ms
        )
        estimates.append(Estimate(dp, finish_ms, read, queue_ms, compute_ms))

    feasible = apply_health_load_network_slo_guards(estimates, snapshot)
    if read_time_spread(feasible) < read_noise_threshold(feasible):
        selected = min(feasible, key=lambda item: item.queue_ms + item.compute_ms)
    else:
        selected = stable_min_with_tie_rotation(feasible)

    authoritative = scheduler_server.recheck(selected, estimates)
    scheduler_server.reserve(
        authoritative.dp,
        authoritative.compute_ms,
        kv_plan.source_node,
        kv_plan.planned_bytes,
    )
    return authoritative.dp
```

## 10. 模块设计建议

```text
motor/coordinator/scheduler/
├── policy/
│   └── kv_fetch_cost_balance.py
├── predictor/
│   ├── kv_read_predictor.py
│   ├── compute_time_predictor.py
│   ├── overlap_predictor.py
│   └── quantile_bucket.py
├── telemetry/
│   ├── kv_read_event.py
│   └── scheduler_snapshot.py
└── runtime/
    ├── scheduler_client.py
    └── scheduler_server.py
```

职责划分：

- `KVReadPredictor`：输出候选 DP 的读取 P50/P90 和置信度；
- `ComputeTimePredictor`：输出 DP 排队和请求计算时间；
- `OverlapPredictor`：估计读取与计算重叠；
- `KVFetchCostBalancePolicy`：候选过滤和统一评分；
- `SchedulerClient`：准备请求及 KV 元数据，提出候选；
- `SchedulerServer`：使用权威状态重新评分、预留负载并提交分配；
- `Telemetry`：接收实际执行事件并更新在线模型。

## 11. 配置建议

```json
{
  "motor_coordinator_config": {
    "scheduler_config": {
      "scheduler_type": "kv_fetch_cost_balance",
      "kv_read_risk_weight": 0.5,
      "kv_read_noise_threshold_ms": 2.0,
      "compute_overload_threshold": 1.5,
      "network_utilization_threshold": 0.85,
      "min_switch_improvement": 0.05,
      "migration_cooldown_seconds": 30,
      "predictor_ewma_alpha": 0.2,
      "predictor_min_samples": 20,
      "exploration_ratio": 0.01
    }
  }
}
```

配置原则：

- 默认值应通过离线 trace 回放和小流量灰度确定；
- 所有阈值支持动态更新；
- `exploration_ratio` 应保持较小，仅用于刷新冷门 DP 的估计；
- 无全局池或缺少读取遥测时自动降级到现有负载均衡策略。

## 12. 降级与容错

| 异常场景 | 降级行为 |
|----------|----------|
| KV 查询失败 | 按完整 prompt 计算，使用计算均衡 |
| 读取预测无样本 | 使用拓扑理论带宽和保守固定开销 |
| 读取预测置信度低 | 增大风险项或忽略 DP 间小差异 |
| 计算状态过期 | Scheduler Server 刷新快照后重新仲裁 |
| 所有候选超过负载阈值 | 选择预计完成时间最小的候选 |
| 所有候选违反 SLO | 选择预计 SLO 超时最小的候选 |
| 网络指标不可用 | 不使用网络外部性项，保留读取历史分位数 |
| 在线模型误差持续超限 | 降级到 Power-of-Two + 最短虚拟计算队列 |
| 预留状态泄漏 | 使用请求完成、超时和定期对账三种机制回收 |

## 13. 评估方案

### 13.1 预测准确性

- KV 读取 P50/P90 绝对误差和相对误差；
- 计算时间 P50/P90 预测误差；
- 分位数覆盖率，例如预测 P90 应覆盖约 90% 样本；
- 按 KV 大小、拓扑和并发度分桶检查误差；
- 模型冷启动到稳定所需样本数。

### 13.2 调度效果

- 满足 SLO 的 prompt TPS 和总 TPS；
- TTFT P50/P90/P99；
- KV 读取时间 P50/P90/P99；
- DP 计算负载 `max/mean`、变异系数和空闲比例；
- 源节点、目标 NIC 的利用率与排队字节；
- 会话迁移率和无收益迁移比例；
- Scheduler 决策耗时；
- 预测器降级和服务端重选比例。

### 13.3 对比策略

- Round Robin；
- 仅计算负载均衡；
- 仅 KV 命中率调度；
- 现有 KV Cache Affinity；
- SMetric 会话中心调度；
- 本文策略关闭读取计算重叠后的消融版本；
- 本文策略关闭风险项和网络外部性后的消融版本。

### 13.4 验收标准建议

- 调度器单次评分 P99 耗时不超过配置预算；
- 读取 P90 预测覆盖率处于可接受区间；
- DP 计算负载 `max/mean` 不劣于现有负载均衡策略设定上限；
- 在读取成本存在明显拓扑差异时，TTFT P90/P99 优于纯计算均衡；
- 在读取成本无显著差异时，吞吐和负载均衡不劣于纯计算均衡；
- 模型不可用时能够自动降级且不影响请求正确性。

## 14. 分阶段落地

### 阶段一：可观测性

- 增加 KV 读取分段埋点；
- 建立 DP、源节点和网络状态快照；
- 仅离线记录预测结果，不影响线上路由。

### 阶段二：影子预测

- 部署解析模型和分桶 EWMA；
- 对比预测与真实读取、计算时间；
- 完成冷启动、异常值和分位数校准。

### 阶段三：候选集内灰度

- 仅在读取时间差超过噪声阈值时启用新策略；
- 保留服务端过载保护和现有负载均衡降级；
- 从低流量租户开始灰度。

### 阶段四：全量与自适应

- 引入网络外部性影子价格；
- 动态调整风险系数和过载阈值；
- 根据 SLO、吞吐和预测误差自动启停读取成本项。

## 15. 风险与待确认项

1. KV Pool 和推理引擎是否能够提供计划读取字节、源节点及分段时间；
2. Layerwise KV 传输与 prefill 计算的真实重叠方式；
3. 多 Scheduler Client 下资源预留的一致性和回收机制；
4. 共置模式中 Decode 对 Prefill 的干扰模型；
5. KV 复制、压缩和动态迁移是否会改变来源与传输字节；
6. 调度预测本身的 CPU 开销和高并发扩展性；
7. 不同模型和硬件是否需要独立模型参数；
8. SLO 目标是 TTFT 优先、吞吐优先还是多目标约束。

在上述接口尚未确认前，不建议直接使用固定权重把 KV 大小与现有负载分数
相加。优先完成时间分段埋点和影子预测，再依据真实误差决定生产调度模型。
