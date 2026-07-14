# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Balanced session-centric scheduling for agentic requests."""

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import ConductorApiClient, TENANT_ID
from motor.coordinator.models.constants import OpenAIField
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.kv_cache_affinity import (
    KvCacheAffinityPolicy,
    TokenizerManager,
)
from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy


logger = get_logger(__name__)


class SMetricPolicy(KvCacheAffinityPolicy):
    """Balance new sessions and preserve local KV affinity for later turns."""

    @staticmethod
    def select_session_candidates_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
        role: PDRole,
        overload_threshold: float = 2.0,
        hit_ratio: float = 0.5,
        top_k: int = 1,
        instance_score_weight: float = 0.05,
        start_index: int = 0,
    ) -> tuple[list[tuple[Instance, Endpoint, float]], bool]:
        """Return candidates and whether the result uses session affinity.

        A first-turn request, an evicted session, or an overloaded affinity target
        uses global load balancing. A healthy follow-up request sticks to the
        endpoint with the longest cached prefix.
        """
        load_balanced = SMetricPolicy._load_balance_candidates(
            instances,
            role,
            top_k,
            instance_score_weight,
            start_index,
        )
        if not SMetricPolicy._is_followup_request(req_info):
            return load_balanced, False

        encoded_ids = KvCacheAffinityPolicy._ensure_token_ids(req_info)
        expected_hit = SMetricPolicy._estimate_history_tokens(req_info, encoded_ids)
        if expected_hit <= 0:
            return load_balanced, False

        block_size = KvCacheAffinityPolicy._conductor_block_size()
        if block_size > 0 and len(encoded_ids) < block_size:
            return load_balanced, False

        response = ConductorApiClient.query_conductor(instances, encoded_ids)
        tenant = response.get(TENANT_ID)
        if tenant is None:
            logger.warning("smetric: conductor tenant unavailable; using load balance")
            return load_balanced, False

        raw, any_instance = KvCacheAffinityPolicy._collect_load_candidates(
            instances,
            tenant,
            len(encoded_ids),
            overlap_credit=1.0,
        )
        if not any_instance or not raw:
            return load_balanced, False

        load_cost, matched_tokens, _prefill, instance, endpoint = min(
            raw,
            key=lambda candidate: (-candidate[1], candidate[0]),
        )
        if matched_tokens < max(0.0, hit_ratio) * expected_hit:
            logger.debug(
                "smetric: expected session prefix was evicted (matched=%s expected=%s ratio=%.2f)",
                matched_tokens,
                expected_hit,
                hit_ratio,
            )
            return load_balanced, False

        mean_load = sum(candidate[0] for candidate in raw) / len(raw)
        threshold = max(1.0, overload_threshold)
        if load_cost > threshold * mean_load:
            logger.debug(
                "smetric: affinity target overloaded (load=%.2f mean=%.2f threshold=%.2f)",
                load_cost,
                mean_load,
                threshold,
            )
            return load_balanced, False

        logger.info(
            "smetric: stick follow-up to %s-%s matched=%s expected=%s load=%.2f",
            instance.id,
            endpoint.id,
            matched_tokens,
            expected_hit,
            load_cost,
        )
        return [(instance, endpoint, -float(matched_tokens))], True

    @staticmethod
    def _load_balance_candidates(
        instances: list[Instance],
        role: PDRole,
        top_k: int,
        instance_score_weight: float,
        start_index: int,
    ) -> list[tuple[Instance, Endpoint, float]]:
        candidates = LoadBalancePolicy.select_endpoint_candidates_from_list(
            instances,
            role,
            top_k=max(1, top_k),
            instance_score_weight=instance_score_weight,
            start_index=start_index,
        )
        return [(candidate.instance, candidate.endpoint, candidate.score) for candidate in candidates]

    @staticmethod
    def _is_followup_request(req_info: RequestInfo) -> bool:
        request_data = req_info.req_data
        explicit_turn = request_data.get("session_turn")
        if explicit_turn is not None:
            try:
                return int(explicit_turn) > 0
            except (TypeError, ValueError):
                logger.warning("smetric: invalid session_turn=%r; inferring from messages", explicit_turn)

        messages = request_data.get(OpenAIField.MESSAGES)
        if not isinstance(messages, list):
            return False
        return any(isinstance(message, dict) and message.get("role") in {"assistant", "tool"} for message in messages)

    @staticmethod
    def _estimate_history_tokens(
        req_info: RequestInfo,
        encoded_ids: list[int],
    ) -> int:
        request_data = req_info.req_data
        messages = request_data.get(OpenAIField.MESSAGES)
        if isinstance(messages, list):
            last_assistant = next(
                (
                    index
                    for index in range(len(messages) - 1, -1, -1)
                    if isinstance(messages[index], dict) and messages[index].get("role") == "assistant"
                ),
                -1,
            )
            if last_assistant >= 0:
                history = messages[: last_assistant + 1]
                tools = request_data.get(OpenAIField.TOOLS)
                return len(TokenizerManager().apply_chat_template(history, tools))

        explicit_turn = request_data.get("session_turn")
        try:
            if explicit_turn is not None and int(explicit_turn) > 0:
                return len(encoded_ids)
        except (TypeError, ValueError):
            pass
        return 0
