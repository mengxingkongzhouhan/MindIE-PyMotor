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

"""
Lightweight per-DP engine queue depth cache for Scheduler allocate/select logs.

Scrapes only vllm:num_requests_running / vllm:num_requests_waiting from each
endpoint's mgmt /metrics. Endpoint id is the DP rank.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass

from motor.common.logger import get_logger
from motor.common.resources.instance import PDRole
from motor.coordinator.api_client.engine_server_api_client import EngineServerApiClient
from motor.coordinator.domain.instance_manager import InstanceManager

logger = get_logger(__name__)

# Sum all series for the gauge on one endpoint's /metrics text (one DP).
_RUNNING_RE = re.compile(
    r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)
_WAITING_RE = re.compile(
    r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)

_QUEUE_ROLES = (PDRole.ROLE_P, PDRole.ROLE_D, PDRole.ROLE_U, PDRole.ROLE_E)


@dataclass(frozen=True)
class EndpointQueueStats:
    """Engine-reported queue depths for one DP endpoint."""

    running: int
    waiting: int


def parse_queue_stats_from_metrics(metrics_text: str) -> EndpointQueueStats | None:
    """
    Extract running/waiting request counts from Prometheus text.

    Returns None when neither gauge is present. Missing one gauge is treated as 0
    when the other is present.
    """
    if not metrics_text:
        return None
    running_vals = [float(m.group(1)) for m in _RUNNING_RE.finditer(metrics_text)]
    waiting_vals = [float(m.group(1)) for m in _WAITING_RE.finditer(metrics_text)]
    if not running_vals and not waiting_vals:
        return None
    return EndpointQueueStats(
        running=int(sum(running_vals)),
        waiting=int(sum(waiting_vals)),
    )


def format_role_queue_stats(stats: dict[str, EndpointQueueStats]) -> str:
    """Format per-DP queue depths for logs: ins:dp=rN/wM,..."""
    if not stats:
        return "none"
    return ",".join(
        f"{key}=r{value.running}/w{value.waiting}" for key, value in stats.items()
    )


class EndpointQueueStatsCache:
    """
    Thread-safe cache of latest engine running/waiting counts per (instance, endpoint).

    Refreshed periodically by SchedulerServer; ALLOCATE_ONLY only reads the cache.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stats: dict[tuple[int, int], EndpointQueueStats] = {}

    def snapshot(self) -> dict[tuple[int, int], EndpointQueueStats]:
        with self._lock:
            return dict(self._stats)

    def get(self, instance_id: int, endpoint_id: int) -> EndpointQueueStats | None:
        with self._lock:
            return self._stats.get((instance_id, endpoint_id))

    def role_stats(
        self,
        instance_manager: InstanceManager,
        role: PDRole,
    ) -> dict[str, EndpointQueueStats]:
        """Return {\"ins:dp\": EndpointQueueStats} for endpoints in the role pool."""
        snap = self.snapshot()
        out: dict[str, EndpointQueueStats] = {}
        for instance in instance_manager.get_available_instances(role).values():
            for pod_eps in (instance.endpoints or {}).values():
                for ep in (pod_eps or {}).values():
                    key = f"{instance.id}:{ep.id}"
                    cached = snap.get((instance.id, ep.id))
                    if cached is not None:
                        out[key] = cached
                    else:
                        out[key] = EndpointQueueStats(running=-1, waiting=-1)
        return out

    def replace_all(self, stats: dict[tuple[int, int], EndpointQueueStats]) -> None:
        with self._lock:
            self._stats = dict(stats)

    def refresh_sync(self, instance_manager: InstanceManager) -> int:
        """
        Scrape all available endpoints and replace the cache.

        Returns number of endpoints successfully updated.
        """
        targets: list[tuple[int, int, str, str]] = []
        for role in _QUEUE_ROLES:
            for instance in instance_manager.get_available_instances(role).values():
                for ep in instance.get_all_endpoints():
                    if not ep.ip or not ep.mgmt_port:
                        continue
                    targets.append((instance.id, ep.id, ep.ip, ep.mgmt_port))

        updated: dict[tuple[int, int], EndpointQueueStats] = {}
        ok = 0
        for instance_id, endpoint_id, ip, mgmt_port in targets:
            metrics_text = EngineServerApiClient.query_metrics(f"{ip}:{mgmt_port}")
            parsed = parse_queue_stats_from_metrics(metrics_text or "")
            if parsed is None:
                continue
            updated[(instance_id, endpoint_id)] = parsed
            ok += 1

        # Preserve previous values for endpoints that failed this round.
        with self._lock:
            merged = dict(self._stats)
            merged.update(updated)
            # Drop endpoints that no longer exist.
            live_keys = {(iid, eid) for iid, eid, _, _ in targets}
            self._stats = {k: v for k, v in merged.items() if k in live_keys}
        return ok

    async def refresh(self, instance_manager: InstanceManager) -> int:
        """Async wrapper: run blocking HTTP scrapes off the event loop."""
        return await asyncio.to_thread(self.refresh_sync, instance_manager)
