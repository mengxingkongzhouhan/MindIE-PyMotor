# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Tests for EndpointQueueStatsCache and Prometheus queue-depth parsing."""

from unittest.mock import patch

import pytest

from motor.common.resources.endpoint import Endpoint, EndpointStatus, Workload
from motor.common.resources.http_msg_spec import EventType
from motor.common.resources.instance import Instance, InsStatus, PDRole, ParallelConfig
from motor.config.coordinator import CoordinatorConfig, SchedulerType
from motor.coordinator.domain.instance_manager import InstanceManager
from motor.coordinator.scheduler.runtime.endpoint_queue_stats import (
    EndpointQueueStats,
    EndpointQueueStatsCache,
    format_role_queue_stats,
    parse_queue_stats_from_metrics,
)
from motor.coordinator.scheduler.runtime.scheduler_server import _SchedulerRequestDispatcher
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    SchedulerRequest,
    SchedulerRequestType,
    SchedulerResponseType,
)
from motor.coordinator.scheduler.scheduler import Scheduler


SAMPLE_METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="/model"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="/model"} 5.0
"""


def test_parse_queue_stats_from_metrics_sums_series():
    text = SAMPLE_METRICS + 'vllm:num_requests_running{engine="1"} 2.0\n'
    stats = parse_queue_stats_from_metrics(text)
    assert stats is not None
    assert stats.running == 5
    assert stats.waiting == 5


def test_parse_queue_stats_from_metrics_missing_returns_none():
    assert parse_queue_stats_from_metrics("") is None
    assert parse_queue_stats_from_metrics("# HELP other\nother 1.0\n") is None


def test_format_role_queue_stats():
    formatted = format_role_queue_stats(
        {
            "1:10": EndpointQueueStats(running=3, waiting=5),
            "1:11": EndpointQueueStats(running=0, waiting=2),
        }
    )
    assert formatted == "1:10=r3/w5,1:11=r0/w2"
    assert format_role_queue_stats({}) == "none"


@pytest.mark.asyncio
async def test_allocate_only_logs_per_dp_running_waiting():
    """ALLOCATE_ONLY response includes per-DP engine running/waiting from the cache."""
    config = CoordinatorConfig()
    config.scheduler_config.scheduler_type = SchedulerType.LOAD_BALANCE
    config.scheduler_config.endpoint_instance_score_weight = 0.0
    instance_manager = InstanceManager(config)

    def _make_instance(instance_id: int, endpoint_ids: tuple[int, int], role: PDRole):
        inst = Instance(
            job_name=f"{role.value}-{instance_id}",
            model_name="test_model",
            id=instance_id,
            role=role,
            status=InsStatus.ACTIVE,
            parallel_config=ParallelConfig(dp_size=2),
        )
        inst.add_endpoints(
            f"pod-{instance_id}",
            {
                idx: Endpoint(
                    id=endpoint_id,
                    ip=f"10.0.0.{instance_id}",
                    business_port=f"80{idx}",
                    mgmt_port=f"90{idx}",
                    status=EndpointStatus.NORMAL,
                    workload=Workload(),
                )
                for idx, endpoint_id in enumerate(endpoint_ids)
            },
        )
        return inst

    prefill = _make_instance(1, (10, 11), PDRole.ROLE_P)
    decode = _make_instance(2, (20, 21), PDRole.ROLE_D)
    await instance_manager.refresh_instances(EventType.ADD, [prefill, decode])

    cache = EndpointQueueStatsCache()
    cache.replace_all(
        {
            (1, 10): EndpointQueueStats(running=3, waiting=5),
            (1, 11): EndpointQueueStats(running=0, waiting=2),
            (2, 20): EndpointQueueStats(running=10, waiting=1),
            (2, 21): EndpointQueueStats(running=4, waiting=0),
        }
    )

    scheduler = Scheduler(instance_provider=instance_manager, config=config)
    dispatcher = _SchedulerRequestDispatcher(
        instance_manager,
        scheduler,
        config,
        queue_stats_cache=cache,
    )
    request = SchedulerRequest(
        request_type=SchedulerRequestType.ALLOCATE_ONLY,
        request_id="alloc-queue",
        data={
            "instance_id": 1,
            "endpoint_id": 10,
            "role": PDRole.ROLE_P.value,
            "req_id": "req-queue",
            "workload": Workload(active_tokens=1, active_requests=1).model_dump(
                mode="json"
            ),
            "workload_sequence": 0,
            "instance_version": 1,
        },
    )

    response = await dispatcher.dispatch(request)

    assert response.response_type == SchedulerResponseType.SUCCESS
    assert response.data["prefill_dps"] == {
        "1:10": {"running": 3, "waiting": 5},
        "1:11": {"running": 0, "waiting": 2},
    }
    assert response.data["decode_dps"] == {
        "2:20": {"running": 10, "waiting": 1},
        "2:21": {"running": 4, "waiting": 0},
    }


def test_queue_stats_cache_refresh_sync():
    """refresh_sync scrapes each endpoint and stores parsed running/waiting."""
    config = CoordinatorConfig()
    instance_manager = InstanceManager(config)
    inst = Instance(
        job_name="prefill-1",
        model_name="m",
        id=1,
        role=PDRole.ROLE_P,
        status=InsStatus.ACTIVE,
        parallel_config=ParallelConfig(dp_size=1),
    )
    inst.add_endpoints(
        "pod-1",
        {
            0: Endpoint(
                id=10,
                ip="10.0.0.1",
                business_port="8080",
                mgmt_port="9090",
                status=EndpointStatus.NORMAL,
            )
        },
    )

    async def _add():
        await instance_manager.refresh_instances(EventType.ADD, [inst])

    import asyncio

    asyncio.run(_add())

    cache = EndpointQueueStatsCache()
    with patch(
        "motor.coordinator.scheduler.runtime.endpoint_queue_stats.EngineServerApiClient.query_metrics",
        return_value=SAMPLE_METRICS,
    ):
        ok = cache.refresh_sync(instance_manager)

    assert ok == 1
    assert cache.get(1, 10) == EndpointQueueStats(running=3, waiting=5)
