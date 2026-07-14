# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
Balanced session-centric scheduling policy for agentic serving.

Implements the differential routing idea from "SMetric: Rethink LLM Scheduling for
Serving Agents with Balanced Session-centric Scheduling" (arXiv:2607.08565): agent
sessions reuse most of their KV$ from the *previous turn on the same instance*
(intra-session locality), so a cache-aware scheduler that always chases the highest
KV$ hit ends up pinning entire sessions -- and therefore the cluster's load -- to
whichever instance served each session's first request. SMetric breaks that coupling
by routing every session's first request purely for load balance (spreading sessions
across the cluster) and only cache-aware-routing follow-up requests (so a session's
later turns keep reusing the local KV$ of the instance that served its previous
turn). Two guards keep the cache-aware path honest: a request never sticks to an
overloaded instance, and never sticks to a session whose local KV$ was likely
evicted.

The session turn is derived from the request's own carried conversation history
(no per-session state kept by the router), so this policy stays as stateless as
LoadBalance/KvCacheAffinity: two requests for the same session never need to be
seen by the same router process for this policy to behave correctly.
"""

from __future__ import annotations

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint, Workload, WorkloadAction
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import ConductorApiClient, TENANT_ID
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.models.constants import OpenAIField
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy
from motor.coordinator.scheduler.policy.kv_cache_affinity import KvCacheAffinityPolicy, TokenizerManager
from motor.coordinator.scheduler.policy.load_balance import LoadBalancePolicy

logger = get_logger(__name__)

# Defaults mirror SchedulerConfig.session_affinity_* (motor/config/coordinator.py); kept here so
# the static ranking methods have sane standalone defaults independent of CoordinatorConfig.
DEFAULT_OVERLAP_CREDIT = 1.0
DEFAULT_OVERLOAD_FACTOR = 2.0
DEFAULT_HIT_RATIO = 0.7

_ASSISTANT_ROLE = "assistant"


def infer_session_turn(req_info: RequestInfo) -> int:
    """
    Infer this request's turn index within its agent session (0 = first turn),
    derived only from the request's own carried history -- no per-session state.

    An agent session appends exactly one assistant reply to the conversation for
    every completed turn (see Figure 3 of SMetric): request 1 has no assistant
    message yet, request 2 carries the first assistant reply, and so on. Counting
    assistant-role messages already in the request's ``messages`` therefore gives
    the number of turns this session has completed so far. A prompt-only request
    (``/v1/completions``, no ``messages``) or a fresh chat with no assistant turn
    yet is turn 0, matching a brand-new session.
    """
    messages = (req_info.req_data or {}).get(OpenAIField.MESSAGES, None)
    if not messages:
        return 0
    return sum(1 for m in messages if isinstance(m, dict) and m.get(OpenAIField.ROLE) == _ASSISTANT_ROLE)


class SessionAffinityPolicy(BaseSchedulingPolicy):
    """
    SMetric-style balanced session-centric scheduling policy.

    Like :class:`KvCacheAffinityPolicy`, the real ranking lives in the
    ``req_info``-aware static methods below (used by the distributed
    ``AsyncSchedulerClient``); ``_select_instance``/``_select_endpoint`` are
    intentionally no-ops for the in-process, request-unaware ``select_instance_and_endpoint``
    path.
    """

    def __init__(self, instance_provider: InstanceProvider):
        super().__init__(instance_provider=instance_provider)
        self._instance_provider = instance_provider
        logger.info("SessionAffinityPolicy started.")

    @staticmethod
    def select_endpoint_candidates_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
        overlap_credit: float = DEFAULT_OVERLAP_CREDIT,
        overload_factor: float = DEFAULT_OVERLOAD_FACTOR,
        hit_ratio: float = DEFAULT_HIT_RATIO,
        top_k: int = 1,
    ) -> list[tuple[Instance, Endpoint, float]] | None:
        """
        Route by session turn: first-turn requests are load-balanced across the cluster;
        follow-up requests stick to the instance holding the most local KV$ for this session,
        unless a guard trips, in which case they fall back to the same load-balanced routing
        as a first-turn request (so the cluster stays balanced even in the tail case where
        stickiness would otherwise overload an instance or chase an already-evicted cache).

        Returns up to ``top_k`` best-first ``(instance, endpoint, score)`` tuples, or ``None``
        when no candidate can be ranked at all (e.g. no instances). Never returns ``None``
        merely because the follow-up guards tripped -- that case still returns the
        load-balanced candidates.

        :param instances: candidate prefill instances.
        :param req_info: request whose turn/prompt drive routing.
        :param overlap_credit: how much a cached prefix discounts prefill work when ranking
            follow-up requests by local KV$ hit (default 1.0).
        :param overload_factor: follow-up requests only stick to their session's instance while
            its load is <= ``overload_factor`` times the mean cluster load (default 2.0); must
            be > 1 to allow any stickiness at all.
        :param hit_ratio: follow-up requests only stick when the best actual local KV$ hit is
            >= ``hit_ratio`` times the hit estimated from the request's own carried history
            (default 0.7); guards against sticking to a session whose cache was evicted.
        :param top_k: maximum number of ranked candidates to return (>=1).
        """
        if infer_session_turn(req_info) == 0:
            return SessionAffinityPolicy._select_balanced(instances, top_k)

        sticky = SessionAffinityPolicy._select_follow_up(
            instances, req_info, overlap_credit, overload_factor, hit_ratio, top_k
        )
        if sticky is not None:
            return sticky
        # A guard tripped (overloaded instance / likely-evicted session cache) or the conductor
        # had no signal: fall back to the same balanced routing a first-turn request would get,
        # rather than silently degrading to round-robin.
        return SessionAffinityPolicy._select_balanced(instances, top_k)

    @staticmethod
    def select_endpoint_from_list(
        instances: list[Instance],
        req_info: RequestInfo,
        overlap_credit: float = DEFAULT_OVERLAP_CREDIT,
        overload_factor: float = DEFAULT_OVERLOAD_FACTOR,
        hit_ratio: float = DEFAULT_HIT_RATIO,
    ) -> tuple[Instance, Endpoint] | None:
        """Single-result convenience wrapper over :meth:`select_endpoint_candidates_from_list`."""
        ranked = SessionAffinityPolicy.select_endpoint_candidates_from_list(
            instances,
            req_info,
            overlap_credit=overlap_credit,
            overload_factor=overload_factor,
            hit_ratio=hit_ratio,
            top_k=1,
        )
        if not ranked:
            return None
        instance, endpoint, _score = ranked[0]
        return (instance, endpoint)

    @staticmethod
    def _select_balanced(instances: list[Instance], top_k: int = 1) -> list[tuple[Instance, Endpoint, float]] | None:
        """First-turn (and guard-tripped) routing: spread sessions across the cluster by load."""
        candidates = LoadBalancePolicy.select_endpoint_candidates_from_list(instances, role=PDRole.ROLE_P, top_k=top_k)
        if not candidates:
            return None
        return [(c.instance, c.endpoint, c.score) for c in candidates]

    @staticmethod
    def _select_follow_up(
        instances: list[Instance],
        req_info: RequestInfo,
        overlap_credit: float,
        overload_factor: float,
        hit_ratio: float,
        top_k: int,
    ) -> list[tuple[Instance, Endpoint, float]] | None:
        """
        Rank candidates by highest local KV$ hit (session stickiness), then gate the winner by
        the two SMetric guards. Returns ``None`` when there is no conductor signal to rank on,
        or when a guard trips (both cases mean "let the caller fall back to load balance").
        """
        encoded_ids = KvCacheAffinityPolicy._ensure_token_ids(req_info)
        block_size = KvCacheAffinityPolicy._conductor_block_size()
        if block_size > 0 and len(encoded_ids) < block_size:
            # Sub-block prompt: can never hit a cached prefix; nothing to stick to.
            return None
        rsp = ConductorApiClient.query_conductor(instances, encoded_ids)
        tenant = rsp.get(TENANT_ID, None)
        if tenant is None:
            return None

        raw, any_instance = KvCacheAffinityPolicy._collect_load_candidates(
            instances, tenant, len(encoded_ids), overlap_credit
        )
        if not any_instance or not raw:
            return None

        # Highest cached prefix first (session stickiness); tie -> lighter load.
        ranked_raw = sorted(raw, key=lambda c: (-c[1], c[0]))
        _top_load, top_matched, _top_prefill, top_instance, _top_ep = ranked_raw[0]

        if not SessionAffinityPolicy._not_overloaded(top_instance, instances, overload_factor):
            logger.debug(
                "session_affinity: instance %s overloaded, falling back to load balance",
                top_instance.id,
            )
            return None
        if not SessionAffinityPolicy._session_not_evicted(req_info, top_matched, hit_ratio):
            logger.debug(
                "session_affinity: matched=%s below hit_ratio guard, falling back to load balance",
                top_matched,
            )
            return None

        ranked = ranked_raw[: max(1, top_k)]
        return [(inst, ep, load_cost) for (load_cost, _m, _p, inst, ep) in ranked]

    @staticmethod
    def _not_overloaded(target_instance: Instance, instances: list[Instance], overload_factor: float) -> bool:
        """
        "not_overloaded" guard (Fig. 13 of SMetric): only stick to ``target_instance`` while its
        load stays within ``overload_factor`` times the mean load across the candidate cluster.
        """
        if overload_factor <= 0:
            return True
        loads = [inst.gathered_workload.calculate_workload_score(PDRole.ROLE_P) for inst in instances]
        if not loads:
            return True
        mean_load = sum(loads) / len(loads)
        if mean_load <= 0:
            return True
        target_load = target_instance.gathered_workload.calculate_workload_score(PDRole.ROLE_P)
        return target_load <= overload_factor * mean_load

    @staticmethod
    def _session_not_evicted(req_info: RequestInfo, matched_tokens: int, hit_ratio: float) -> bool:
        """
        "session_not_evicted" guard (Fig. 13 of SMetric): only stick when the best actual local
        hit reaches at least ``hit_ratio`` of the hit expected from the request's own carried
        history. A much lower actual hit means the local KV$ was likely evicted, so the request
        is better treated as a fresh session (load-balanced) than pinned to a cold instance.
        """
        if hit_ratio <= 0:
            return True
        est_hit = SessionAffinityPolicy._estimate_history_hit_len(req_info)
        if est_hit <= 0:
            # Can't form an expectation (e.g. no prior assistant turn to exclude); don't block.
            return True
        return matched_tokens >= hit_ratio * est_hit

    @staticmethod
    def _estimate_history_hit_len(req_info: RequestInfo) -> int:
        """
        Tokens expected to already be cached: the request's carried history *excluding* the
        freshly appended turn, which by definition was never seen by any instance before.
        """
        messages = (req_info.req_data or {}).get(OpenAIField.MESSAGES, None)
        if not messages or len(messages) < 2:
            return 0
        tools = req_info.req_data.get(OpenAIField.TOOLS, None)
        try:
            return len(TokenizerManager().apply_chat_template(messages[:-1], tools))
        except Exception as e:  # pragma: no cover - defensive, mirrors KvCacheAffinityPolicy
            logger.debug("session_affinity: could not estimate history hit length: %s", e)
            return 0

    def _select_instance(self, _: PDRole = None) -> Instance | None:
        return None

    def _select_endpoint(self, _: Instance) -> Endpoint | None:
        return None

    async def update_workload(
        self,
        instance_id: int,
        endpoint_id: int,
        req_id: str,
        workload_action: WorkloadAction,
        workload_change: Workload,
    ) -> bool:
        """
        Update workload after session-affinity selection; the central ledger is still needed by
        decode/fallback load-balance paths and worker SHM synchronization (mirrors
        KvCacheAffinityPolicy.update_workload).
        """
        if hasattr(self._instance_provider, "update_instance_workload"):
            await self._instance_provider.update_instance_workload(instance_id, endpoint_id, workload_change)
        else:
            raise RuntimeError("InstanceProvider must support update_instance_workload for SessionAffinityPolicy")

        logger.debug(
            "Request %s updated workload: instance_id=%s, endpoint_id=%s, action=%s, change=%s",
            req_id or "",
            instance_id,
            endpoint_id,
            workload_action.value,
            workload_change,
        )
        return True
