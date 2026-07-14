# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 license for more details.

"""Tests for SessionAffinityPolicy (SMetric-style balanced session-centric scheduling)."""

import unittest
from unittest.mock import Mock, patch

from motor.coordinator.api_client.conductor_api_client import TENANT_ID
from motor.coordinator.scheduler.policy.session_affinity import (
    SessionAffinityPolicy,
    infer_session_turn,
)
from motor.common.resources.endpoint import Endpoint, Workload


def _make_endpoint(ep_id: int, active_tokens: float = 0.0, active_kv_cache: float = 0.0) -> Endpoint:
    """Build a real Endpoint carrying a known workload for load-aware scoring tests."""
    return Endpoint(
        id=ep_id,
        ip="127.0.0.1",
        business_port="8000",
        mgmt_port="8001",
        workload=Workload(active_tokens=active_tokens, active_kv_cache=active_kv_cache),
    )


def _make_instance(instance_id, endpoints, gathered_active_tokens: float = 0.0):
    """Build a mock Instance whose gathered_workload sums the given endpoints' active_tokens."""
    inst = Mock()
    inst.id = instance_id
    inst.endpoints = {"group": {ep.id: ep for ep in endpoints}}
    inst.get_all_endpoints.return_value = tuple(endpoints)
    inst.gathered_workload = Workload(active_tokens=gathered_active_tokens)
    return inst


def _first_turn_req(prompt: str = "hello") -> Mock:
    """A request whose message history has no assistant reply yet -> turn 0."""
    req = Mock()
    req.req_data = {"messages": [{"role": "user", "content": prompt}]}
    req.token_ids = None
    return req


def _follow_up_req(prompt: str = "then what?") -> Mock:
    """A request whose history already carries one completed turn -> turn 1."""
    req = Mock()
    req.req_data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi, how can I help?"},
            {"role": "user", "content": prompt},
        ]
    }
    req.token_ids = None
    return req


def _patch_tokenizers(full_history_len: int = 200, prior_history_len: int | None = None):
    """
    Patch both places that call ``TokenizerManager()``: ``kv_cache_affinity`` (used by
    ``_ensure_token_ids`` to tokenize the *full* prompt, driving ``isl``/conductor query) and
    ``session_affinity`` (used by ``_estimate_history_hit_len`` to tokenize the history *minus*
    the freshly appended turn). Returns the two ``unittest.mock.patch`` context managers.
    """
    if prior_history_len is None:
        prior_history_len = full_history_len
    full_ids = list(range(full_history_len))
    prior_ids = list(range(prior_history_len))

    kva_tokenizer = Mock()
    kva_tokenizer.apply_chat_template.return_value = full_ids
    kva_patch = patch(
        "motor.coordinator.scheduler.policy.kv_cache_affinity.TokenizerManager",
        return_value=kva_tokenizer,
    )

    sa_tokenizer = Mock()
    sa_tokenizer.apply_chat_template.return_value = prior_ids
    sa_patch = patch(
        "motor.coordinator.scheduler.policy.session_affinity.TokenizerManager",
        return_value=sa_tokenizer,
    )
    return kva_patch, sa_patch


class TestInferSessionTurn(unittest.TestCase):
    """infer_session_turn derives the turn purely from the request's own history."""

    def test_prompt_only_request_is_turn_zero(self):
        req = Mock()
        req.req_data = {"prompt": "hello"}
        self.assertEqual(infer_session_turn(req), 0)

    def test_fresh_chat_session_is_turn_zero(self):
        req = Mock()
        req.req_data = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(infer_session_turn(req), 0)

    def test_one_completed_turn_is_turn_one(self):
        req = _follow_up_req()
        self.assertEqual(infer_session_turn(req), 1)

    def test_counts_multiple_assistant_replies(self):
        req = Mock()
        req.req_data = {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
                {"role": "user", "content": "e"},
            ]
        }
        self.assertEqual(infer_session_turn(req), 2)

    def test_non_dict_messages_are_ignored(self):
        req = Mock()
        req.req_data = {"messages": ["not-a-dict", {"role": "assistant", "content": "x"}]}
        self.assertEqual(infer_session_turn(req), 1)


class TestSessionAffinityPolicyFirstTurn(unittest.TestCase):
    """Turn 0 requests are always routed purely by load, regardless of KV$ signal."""

    def test_first_turn_routes_by_load_without_querying_conductor(self):
        ep_a = _make_endpoint(0, active_tokens=100.0)
        ep_b = _make_endpoint(1, active_tokens=10.0)
        instance = _make_instance("inst", [ep_a, ep_b])

        with patch(
            "motor.coordinator.scheduler.policy.session_affinity.ConductorApiClient.query_conductor"
        ) as mock_query:
            result = SessionAffinityPolicy.select_endpoint_from_list([instance], _first_turn_req())

        mock_query.assert_not_called()
        self.assertIsNotNone(result)
        self.assertEqual(result[0].id, "inst")
        self.assertEqual(result[1].id, 1)  # the less-loaded endpoint

    def test_first_turn_candidates_are_topk_by_load(self):
        ep_a = _make_endpoint(0, active_tokens=100.0)
        ep_b = _make_endpoint(1, active_tokens=10.0)
        ep_c = _make_endpoint(2, active_tokens=50.0)
        instance = _make_instance("inst", [ep_a, ep_b, ep_c])

        ranked = SessionAffinityPolicy.select_endpoint_candidates_from_list([instance], _first_turn_req(), top_k=2)

        self.assertIsNotNone(ranked)
        self.assertEqual([ep.id for _inst, ep, _score in ranked], [1, 2])

    def test_no_instances_returns_none(self):
        result = SessionAffinityPolicy.select_endpoint_from_list([], _first_turn_req())
        self.assertIsNone(result)


class TestSessionAffinityPolicyFollowUp(unittest.TestCase):
    """Turn > 0 requests stick to the highest local KV$ hit, subject to the two guards."""

    @patch("motor.coordinator.scheduler.policy.session_affinity.ConductorApiClient.query_conductor")
    def test_follow_up_sticks_to_highest_local_hit(self, mock_query):
        # inst_a has the cached session (high match, moderate load); inst_b is idle but cold.
        ep_a = _make_endpoint(0, active_tokens=20.0)
        ep_b = _make_endpoint(1, active_tokens=0.0)
        inst_a = _make_instance("inst-a", [ep_a], gathered_active_tokens=20.0)
        inst_b = _make_instance("inst-b", [ep_b], gathered_active_tokens=0.0)

        mock_query.return_value = {
            TENANT_ID: {
                "vllm-prefill-inst-a": {"DP": {"0": 180}},
                "vllm-prefill-inst-b": {"DP": {"1": 0}},
            }
        }

        kva_patch, sa_patch = _patch_tokenizers()
        with kva_patch, sa_patch:
            result = SessionAffinityPolicy.select_endpoint_from_list([inst_a, inst_b], _follow_up_req())

        self.assertIsNotNone(result)
        self.assertEqual(result[0].id, "inst-a")
        self.assertEqual(result[1].id, 0)

    @patch("motor.coordinator.scheduler.policy.session_affinity.ConductorApiClient.query_conductor")
    def test_overloaded_sticky_instance_falls_back_to_load_balance(self, mock_query):
        """not_overloaded guard: a session pinned to a hot instance rebalances instead."""
        ep_a = _make_endpoint(0, active_tokens=500.0)  # very hot: holds the cached session
        ep_b = _make_endpoint(1, active_tokens=1.0)  # idle, but cold
        inst_a = _make_instance("inst-a", [ep_a], gathered_active_tokens=500.0)
        inst_b = _make_instance("inst-b", [ep_b], gathered_active_tokens=1.0)

        mock_query.return_value = {
            TENANT_ID: {
                "vllm-prefill-inst-a": {"DP": {"0": 180}},
                "vllm-prefill-inst-b": {"DP": {"1": 0}},
            }
        }

        kva_patch, sa_patch = _patch_tokenizers()
        with kva_patch, sa_patch:
            result = SessionAffinityPolicy.select_endpoint_from_list(
                [inst_a, inst_b], _follow_up_req(), overload_factor=1.5
            )

        # inst-a has by far the best (only) cached prefix (well above the hit_ratio guard), so
        # only the overload guard is exercised here: its load (500) exceeds 1.5x the mean
        # (250.5) -> guard trips -> load balance picks the far lighter inst_b instead of
        # sticking to the overloaded inst_a.
        self.assertIsNotNone(result)
        self.assertEqual(result[0].id, "inst-b")

    @patch("motor.coordinator.scheduler.policy.session_affinity.ConductorApiClient.query_conductor")
    def test_evicted_session_falls_back_to_load_balance(self, mock_query):
        """session_not_evicted guard: a much-lower-than-expected hit rebalances instead of sticking."""
        ep_a = _make_endpoint(0, active_tokens=5.0)
        ep_b = _make_endpoint(1, active_tokens=1.0)
        inst_a = _make_instance("inst-a", [ep_a], gathered_active_tokens=5.0)
        inst_b = _make_instance("inst-b", [ep_b], gathered_active_tokens=1.0)

        # The conductor reports only a 5-token hit on the (only matching) instance, far below
        # the 100 tokens expected from the request's own carried history: the session's local
        # cache was likely evicted.
        mock_query.return_value = {
            TENANT_ID: {
                "vllm-prefill-inst-a": {"DP": {"0": 5}},
                "vllm-prefill-inst-b": {"DP": {"1": 0}},
            }
        }

        kva_patch, sa_patch = _patch_tokenizers()
        with kva_patch, sa_patch:
            result = SessionAffinityPolicy.select_endpoint_from_list([inst_a, inst_b], _follow_up_req(), hit_ratio=0.7)

        # Falls back to load balance -> the lighter inst_b wins instead of sticking to inst_a.
        self.assertIsNotNone(result)
        self.assertEqual(result[0].id, "inst-b")

    @patch("motor.coordinator.scheduler.policy.session_affinity.ConductorApiClient.query_conductor")
    def test_no_tenant_falls_back_to_load_balance(self, mock_query):
        ep_a = _make_endpoint(0, active_tokens=100.0)
        ep_b = _make_endpoint(1, active_tokens=1.0)
        inst_a = _make_instance("inst-a", [ep_a], gathered_active_tokens=100.0)
        inst_b = _make_instance("inst-b", [ep_b], gathered_active_tokens=1.0)

        mock_query.return_value = {}  # no tenant at all

        kva_patch, sa_patch = _patch_tokenizers()
        with kva_patch, sa_patch:
            result = SessionAffinityPolicy.select_endpoint_from_list([inst_a, inst_b], _follow_up_req())

        self.assertIsNotNone(result)
        self.assertEqual(result[0].id, "inst-b")  # load-balance fallback, not None

    @patch.object(SessionAffinityPolicy, "_select_balanced", return_value=None)
    @patch("motor.coordinator.scheduler.policy.session_affinity.ConductorApiClient.query_conductor")
    def test_returns_none_when_no_instances_at_all(self, mock_query, _mock_balanced):
        """When even the load-balance fallback has nothing to offer, propagate None."""
        kva_patch, sa_patch = _patch_tokenizers()
        with kva_patch, sa_patch:
            result = SessionAffinityPolicy.select_endpoint_candidates_from_list([], _follow_up_req())
        self.assertIsNone(result)

    def test_select_instance_and_endpoint_are_noop(self):
        policy = SessionAffinityPolicy(Mock())
        self.assertIsNone(policy._select_instance())
        self.assertIsNone(policy._select_endpoint(Mock()))


class TestSessionAffinityPolicyUpdateWorkload(unittest.IsolatedAsyncioTestCase):
    """update_workload delegates to the instance provider, mirroring KvCacheAffinityPolicy."""

    async def test_update_workload_delegates_to_instance_provider(self):
        from motor.common.resources.endpoint import WorkloadAction

        provider = Mock()
        provider.update_instance_workload = Mock(return_value=_async_none())
        policy = SessionAffinityPolicy(provider)

        result = await policy.update_workload(1, 2, "req-1", WorkloadAction.ALLOCATION, Workload(active_tokens=5))

        self.assertTrue(result)
        provider.update_instance_workload.assert_called_once()

    async def test_update_workload_raises_without_provider_support(self):
        from motor.common.resources.endpoint import WorkloadAction

        provider = Mock(spec=[])  # no update_instance_workload attribute
        policy = SessionAffinityPolicy(provider)

        with self.assertRaises(RuntimeError):
            await policy.update_workload(1, 2, "req-1", WorkloadAction.ALLOCATION, Workload(active_tokens=5))


async def _async_none():
    return None


class TestSessionAffinityFallbackConsolidation(unittest.TestCase):
    """session_affinity -> load_balance -> round_robin fallback chain in AsyncSchedulerClient."""

    _SESSION_AFFINITY = (
        "motor.coordinator.scheduler.runtime.scheduler_client."
        "SessionAffinityPolicy.select_endpoint_candidates_from_list"
    )
    _RR = "motor.coordinator.scheduler.runtime.scheduler_client.RoundRobinPolicy.select_instance_from_list"

    @staticmethod
    def _make_client():
        from motor.coordinator.scheduler.runtime.scheduler_client import (
            AsyncSchedulerClient,
            SchedulerClientConfig,
        )

        return AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="session_affinity"))

    def test_prefill_role_uses_session_affinity_ranking(self):
        from motor.common.resources.instance import PDRole
        from motor.coordinator.scheduler.runtime.zmq_protocol import (
            CANDIDATE_POLICY_SESSION_AFFINITY,
        )

        client = self._make_client()
        inst, ep = Mock(), Mock()
        req = _first_turn_req()
        ranked = [(inst, ep, 0.0)]
        with patch(self._SESSION_AFFINITY, return_value=ranked):
            cands, policy = client._select_endpoint_candidates_from_list_with_policy(
                [Mock()], PDRole.ROLE_P, req, top_k=1
            )
        self.assertEqual(policy, CANDIDATE_POLICY_SESSION_AFFINITY)
        self.assertEqual(cands, ranked)

    def test_no_ranking_falls_back_to_load_balance(self):
        from motor.common.resources.instance import PDRole
        from motor.coordinator.scheduler.runtime.zmq_protocol import CANDIDATE_POLICY_LOAD_BALANCE

        client = self._make_client()
        inst, ep = Mock(), Mock()
        req = _first_turn_req()
        with (
            patch(self._SESSION_AFFINITY, return_value=None),
            patch.object(
                client,
                "_select_endpoint_candidates_by_load_balance",
                return_value=[(inst, ep, 1.0)],
            ) as lb,
        ):
            cands, policy = client._select_endpoint_candidates_from_list_with_policy(
                [Mock()], PDRole.ROLE_P, req, top_k=1
            )
        self.assertEqual(policy, CANDIDATE_POLICY_LOAD_BALANCE)
        self.assertEqual(cands, [(inst, ep, 1.0)])
        lb.assert_called_once()

    def test_non_prefill_role_skips_session_affinity_ranking(self):
        from motor.common.resources.instance import PDRole
        from motor.coordinator.scheduler.runtime.zmq_protocol import CANDIDATE_POLICY_LOAD_BALANCE

        client = self._make_client()
        inst, ep = Mock(), Mock()
        req = _first_turn_req()
        with (
            patch(self._SESSION_AFFINITY) as ranking,
            patch.object(
                client,
                "_select_endpoint_candidates_by_load_balance",
                return_value=[(inst, ep, 2.0)],
            ),
        ):
            cands, policy = client._select_endpoint_candidates_from_list_with_policy(
                [Mock()], PDRole.ROLE_D, req, top_k=1
            )
        ranking.assert_not_called()
        self.assertEqual(policy, CANDIDATE_POLICY_LOAD_BALANCE)
        self.assertEqual(cands, [(inst, ep, 2.0)])

    def test_load_balance_empty_falls_back_to_round_robin(self):
        from motor.common.resources.instance import PDRole
        from motor.coordinator.scheduler.runtime.zmq_protocol import CANDIDATE_POLICY_ROUND_ROBIN

        client = self._make_client()
        inst, ep = Mock(), Mock()
        req = _first_turn_req()
        with (
            patch(self._SESSION_AFFINITY, return_value=None),
            patch.object(client, "_select_endpoint_candidates_by_load_balance", return_value=[]),
            patch(self._RR, return_value=(inst, 1)),
            patch.object(client, "_select_endpoint_for_instance", return_value=(inst, ep)),
        ):
            cands, policy = client._select_endpoint_candidates_from_list_with_policy(
                [Mock()], PDRole.ROLE_P, req, top_k=1
            )
        self.assertEqual(policy, CANDIDATE_POLICY_ROUND_ROBIN)
        self.assertEqual(cands, [(inst, ep, 0.0)])


if __name__ == "__main__":
    unittest.main()
