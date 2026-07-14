# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Tests for balanced session-centric SMetric scheduling."""

from unittest.mock import Mock, patch

from motor.common.resources.instance import PDRole
from motor.coordinator.api_client.conductor_api_client import TENANT_ID
from motor.coordinator.scheduler.policy.smetric import SMetricPolicy
from motor.coordinator.scheduler.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClientConfig,
)
from tests.coordinator.scheduler.conftest import (
    create_mock_endpoint,
    create_mock_instance,
    create_mock_workload,
)


def _request(messages: list[dict], token_count: int = 200) -> Mock:
    request = Mock()
    request.req_data = {"messages": messages}
    request.token_ids = list(range(token_count))
    return request


def _instances(first_load: float, second_load: float):
    first_endpoint = create_mock_endpoint(10, workload=create_mock_workload(active_tokens=first_load))
    second_endpoint = create_mock_endpoint(20, workload=create_mock_workload(active_tokens=second_load))
    return [
        create_mock_instance(1, endpoints={"pod": {10: first_endpoint}}),
        create_mock_instance(2, endpoints={"pod": {20: second_endpoint}}),
    ]


def _conductor(first_match: int, second_match: int) -> dict:
    return {
        TENANT_ID: {
            "vllm-prefill-1": {"DP": {"10": first_match}},
            "vllm-prefill-2": {"DP": {"20": second_match}},
        }
    }


@patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
def test_first_turn_uses_load_balance_without_cache_query(query_conductor):
    instances = _instances(first_load=10, second_load=1)
    request = _request([{"role": "user", "content": "start"}])

    candidates, uses_affinity = SMetricPolicy.select_session_candidates_from_list(
        instances,
        request,
        PDRole.ROLE_P,
    )

    assert uses_affinity is False
    assert candidates[0][0].id == 2
    query_conductor.assert_not_called()


@patch.object(SMetricPolicy, "_estimate_history_tokens", return_value=100)
@patch.object(SMetricPolicy, "_conductor_block_size", return_value=0)
@patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
def test_followup_sticks_to_resident_session(
    query_conductor,
    _block_size,
    _history_tokens,
):
    instances = _instances(first_load=5, second_load=1)
    request = _request(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
    )
    query_conductor.return_value = _conductor(first_match=90, second_match=10)

    candidates, uses_affinity = SMetricPolicy.select_session_candidates_from_list(
        instances,
        request,
        PDRole.ROLE_P,
        overload_threshold=2.0,
        hit_ratio=0.5,
    )

    assert uses_affinity is True
    assert candidates[0][0].id == 1


@patch.object(SMetricPolicy, "_estimate_history_tokens", return_value=100)
@patch.object(SMetricPolicy, "_conductor_block_size", return_value=0)
@patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
def test_evicted_session_falls_back_to_load_balance(
    query_conductor,
    _block_size,
    _history_tokens,
):
    instances = _instances(first_load=5, second_load=1)
    request = _request(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
    )
    query_conductor.return_value = _conductor(first_match=20, second_match=10)

    candidates, uses_affinity = SMetricPolicy.select_session_candidates_from_list(
        instances,
        request,
        PDRole.ROLE_P,
        hit_ratio=0.5,
    )

    assert uses_affinity is False
    assert candidates[0][0].id == 2


@patch.object(SMetricPolicy, "_estimate_history_tokens", return_value=100)
@patch.object(SMetricPolicy, "_conductor_block_size", return_value=0)
@patch("motor.coordinator.scheduler.policy.smetric.ConductorApiClient.query_conductor")
def test_overloaded_session_target_migrates_by_load(
    query_conductor,
    _block_size,
    _history_tokens,
):
    instances = _instances(first_load=100, second_load=0)
    request = _request(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
    )
    query_conductor.return_value = _conductor(first_match=90, second_match=10)

    candidates, uses_affinity = SMetricPolicy.select_session_candidates_from_list(
        instances,
        request,
        PDRole.ROLE_P,
        overload_threshold=1.5,
        hit_ratio=0.5,
    )

    assert uses_affinity is False
    assert candidates[0][0].id == 2


def test_explicit_session_turn_supports_completion_requests():
    request = Mock()
    request.req_data = {"prompt": "full session prompt", "session_turn": 3}

    assert SMetricPolicy._is_followup_request(request) is True


@patch.object(SMetricPolicy, "select_session_candidates_from_list")
def test_scheduler_client_marks_sticky_candidate_as_smetric(select_candidates):
    instance = _instances(first_load=1, second_load=2)[0]
    endpoint = instance.get_all_endpoints()[0]
    request = _request([{"role": "user", "content": "start"}])
    select_candidates.return_value = ([(instance, endpoint, -100.0)], True)
    client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="smetric"))

    candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
        [instance],
        PDRole.ROLE_P,
        request,
    )

    assert candidates == [(instance, endpoint, -100.0)]
    assert candidate_policy == "smetric"


@patch.object(SMetricPolicy, "select_session_candidates_from_list")
def test_scheduler_client_marks_balanced_candidate_as_load_balance(
    select_candidates,
):
    instance = _instances(first_load=1, second_load=2)[0]
    endpoint = instance.get_all_endpoints()[0]
    request = _request([{"role": "user", "content": "start"}])
    select_candidates.return_value = ([(instance, endpoint, 1.0)], False)
    client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="smetric"))

    candidates, candidate_policy = client._select_endpoint_candidates_from_list_with_policy(
        [instance],
        PDRole.ROLE_P,
        request,
    )

    assert candidates == [(instance, endpoint, 1.0)]
    assert candidate_policy == "load_balance"
