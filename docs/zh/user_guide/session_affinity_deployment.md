# 会话亲和（Session Affinity）调度部署

## 特性介绍

`session_affinity` 是一种面向 Agent（智能体）场景的均衡会话中心调度策略，思路参考自论文
[《SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling》](https://arxiv.org/abs/2607.08565)。

论文观察到：Agent 场景下同一会话（session）的多轮请求会持续复用前一轮所在实例的本地 KV Cache（会话局部性），因此现有的"KV Cache 亲和优先"调度策略（例如 `kv_cache_affinity`）会把整个会话都钉死在其首轮请求落地的实例上——少数实例因此过载，其余实例却空闲，从而拖累集群整体吞吐（TPS）。而单纯的负载均衡调度又会放弃大部分本地 KV Cache 复用收益。

`session_affinity` 用差异化路由化解这一矛盾：

* **会话首轮请求**（`turn == 0`）：完全按负载均衡路由，把不同会话的首轮请求打散到全部实例，从根本上避免集群负载倾斜。
* **会话后续请求**（`turn > 0`）：按本地 KV Cache 命中长度亲和路由，尽量粘滞在已缓存该会话上下文的实例上，从而保留本地 KV Cache 复用收益。

为了避免"粘滞"退化为新的负载倾斜或缓存已失效的假粘滞，`session_affinity` 在后续请求粘滞前还会做两项兜底检查（对齐论文 Figure 13 的 `not_overloaded` / `session_not_evicted` 检查）：

* **过载兜底**（`session_affinity_overload_factor`）：若目标实例负载超过集群平均负载的该倍数，放弃粘滞，改走负载均衡路由。
* **缓存失效兜底**（`session_affinity_hit_ratio`）：若实际命中长度显著低于"按请求自带历史估算应命中的长度"，说明该会话的本地缓存大概率已被淘汰，同样放弃粘滞，改走负载均衡路由。

会话轮次（`turn`）完全从请求自身携带的对话历史（`messages` 中已有多少条 `assistant` 回复）推导得出，Router 不维护任何会话到实例的映射状态，因此该策略与 `load_balance`/`kv_cache_affinity` 一样是无状态（stateless）的。

## 依赖条件

`session_affinity` 复用与 `kv_cache_affinity` 相同的 Mooncake Conductor 基础设施（用于查询各实例上的 KV Cache 前缀命中长度），因此镜像准备、`kv-events-config` 配置等前置条件与 KV Cache 亲和性调度完全一致，请先参考 [KV Cache 亲和部署](KV_cache_affinity_deployment.md) 完成 Mooncake Conductor 相关的镜像与前置配置准备。

## 配置说明

在 `motor_coordinator_config.scheduler_config` 中将 `scheduler_type` 配置为 `session_affinity` 即可启用该策略，并可选配置以下调优参数：

```json
{
  "motor_coordinator_config": {
    "scheduler_config": {
      "scheduler_type": "session_affinity",
      "session_affinity_overlap_credit": 1.0,
      "session_affinity_overload_factor": 2.0,
      "session_affinity_hit_ratio": 0.7
    }
  }
}
```

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `session_affinity_overlap_credit` | `1.0` | 后续请求按本地 KV Cache 命中排序时，缓存前缀对 prefill 计算量的抵扣系数（含义与 `kv_affinity_overlap_credit` 一致）。 |
| `session_affinity_overload_factor` | `2.0` | 过载兜底阈值：目标实例负载超过集群平均负载的该倍数时放弃粘滞，改走负载均衡。需大于 1 才允许任何粘滞。 |
| `session_affinity_hit_ratio` | `0.7` | 缓存失效兜底阈值：实际命中长度低于"按请求自带历史估算应命中长度"的该比例时，认为会话缓存可能已被淘汰，改走负载均衡。 |

其余部署步骤（镜像准备、`kv-events-config` 配置、`deploy.py` 部署流程）与 [KV Cache 亲和部署](KV_cache_affinity_deployment.md) 完全一致，仅需将 `scheduler_type` 替换为 `session_affinity`。
