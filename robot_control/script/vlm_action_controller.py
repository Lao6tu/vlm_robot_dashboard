#!/usr/bin/env python3
"""
VLM Action Motion Controller
===========================
Non-blocking motion controller with separate monitoring channels for VLM
inference output and ultrasonic obstacle triggers.

Priority order:
1) Ultrasonic obstacle trigger -> reverse, random turn, wait for VLM
2) Active path restore after VLM-only stop scan
3) VLM direction, only when ultrasonic is not triggered
4) Near-distance caution -> limit speed / block forward rush
5) Sustained VLM stop -> simple scan turn and path restore
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional
from urllib import error, request


DutyTuple = tuple[int, int, int, int]


def _duties_to_label(duties: DutyTuple) -> str:
    d1, d2, d3, d4 = duties
    if duties == (0, 0, 0, 0):
        return "STOP"
    if d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0:
        return "FORWARD"
    if d1 < 0 and d2 < 0 and d3 > 0 and d4 > 0:
        return "STEER_LEFT"
    if d1 > 0 and d2 > 0 and d3 < 0 and d4 < 0:
        return "STEER_RIGHT"
    return "RAW"


class VLMAction(str, Enum):
    MOVE_FORWARD = "Move Forward"
    SLOW_DOWN = "Slow Down"
    STOP = "Stop"
    STEER_RIGHT = "Steer Right"
    STEER_LEFT = "Steer Left"


def _normalize_action(value: Any) -> VLMAction | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    compact = text.replace("_", " ").replace("-", " ")

    if any(token in compact for token in ("steer right", "turn right", "go right")):
        return VLMAction.STEER_RIGHT
    if any(token in compact for token in ("steer left", "turn left", "go left")):
        return VLMAction.STEER_LEFT
    if any(token in compact for token in ("slow down", "slow", "caution")):
        return VLMAction.SLOW_DOWN
    if any(token in compact for token in ("move forward", "go forward", "forward")):
        return VLMAction.MOVE_FORWARD
    if any(token in compact for token in ("stop", "halt", "brake")):
        return VLMAction.STOP

    return None


def _iter_string_values(obj: Any):
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_string_values(value)
        return
    if isinstance(obj, list):
        for value in obj:
            yield from _iter_string_values(value)


def extract_action_from_result(result: dict[str, Any]) -> VLMAction | None:
    candidate_keys = (
        "action",
        "action_advice",
        "nav_action",
        "motion_action",
        "decision",
        "command",
        "next_action",
    )
    for key in candidate_keys:
        action = _normalize_action(result.get(key))
        if action:
            return action

    for text in _iter_string_values(result):
        action = _normalize_action(text)
        if action:
            return action

    return None


class VLMActionSource:
    """Polls inference /api/status in background and keeps latest parsed action."""

    def __init__(
        self,
        status_url: str,
        poll_interval_sec: float = 0.25,
        timeout_sec: float = 1.0,
    ) -> None:
        self._status_url = status_url
        self._poll_interval_sec = max(0.05, poll_interval_sec)
        self._timeout_sec = max(0.1, timeout_sec)

        self._lock = threading.Lock()
        self._latest_action: VLMAction | None = None
        self._latest_update_mono: float | None = None
        self._latest_error: str | None = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest(self) -> tuple[VLMAction | None, float | None, str | None]:
        with self._lock:
            age = None
            if self._latest_update_mono is not None:
                age = max(0.0, time.monotonic() - self._latest_update_mono)
            return self._latest_action, age, self._latest_error

    def _run(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                action = self._fetch_action_once()
                with self._lock:
                    if action is not None:
                        self._latest_action = action
                    self._latest_update_mono = time.monotonic()
                    self._latest_error = None
            except Exception as exc:
                with self._lock:
                    self._latest_error = str(exc)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._poll_interval_sec - elapsed))

    def _fetch_action_once(self) -> VLMAction | None:
        req = request.Request(
            self._status_url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Inference status request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from inference status API: {exc}") from exc

        result: dict[str, Any] | None = None

        if isinstance(payload, dict):
            latest = payload.get("latest_result")
            if isinstance(latest, dict):
                result = latest
            else:
                result = payload

        if not isinstance(result, dict):
            raise RuntimeError("Inference status payload does not contain latest_result")

        return extract_action_from_result(result)


@dataclass(frozen=True)
class UltrasonicObstacleReading:
    distance_cm: int | None = None
    obstacle_triggered: bool = False
    caution_triggered: bool = False
    age_sec: float | None = None
    error: str | None = None


class UltrasonicObstacleSource:
    """Polls ultrasonic distance in its own channel and exposes trigger state."""

    def __init__(
        self,
        *,
        distance_reader: Callable[[], int | None] | None,
        obstacle_trigger_cm: int,
        caution_cm: int,
        poll_interval_sec: float = 0.1,
    ) -> None:
        self._distance_reader = distance_reader
        self._obstacle_trigger_cm = max(1, obstacle_trigger_cm)
        self._caution_cm = max(self._obstacle_trigger_cm, caution_cm)
        self._poll_interval_sec = max(0.05, poll_interval_sec)

        self._lock = threading.Lock()
        self._latest_distance_cm: int | None = None
        self._latest_update_mono: float | None = None
        self._latest_error: str | None = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running or self._distance_reader is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest(self) -> UltrasonicObstacleReading:
        with self._lock:
            age = None
            if self._latest_update_mono is not None:
                age = max(0.0, time.monotonic() - self._latest_update_mono)
            distance_cm = self._latest_distance_cm
            return UltrasonicObstacleReading(
                distance_cm=distance_cm,
                obstacle_triggered=(
                    distance_cm is not None
                    and distance_cm <= self._obstacle_trigger_cm
                ),
                caution_triggered=(
                    distance_cm is not None and distance_cm <= self._caution_cm
                ),
                age_sec=age,
                error=self._latest_error,
            )

    def _run(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                distance_cm = self._distance_reader() if self._distance_reader else None
                with self._lock:
                    self._latest_distance_cm = distance_cm
                    self._latest_update_mono = time.monotonic()
                    self._latest_error = None
            except Exception as exc:
                with self._lock:
                    self._latest_error = str(exc)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._poll_interval_sec - elapsed))


@dataclass
class MotionPolicy:
    base_speed: int = 1400
    slow_speed: int = 900
    turn_speed: int = 1500
    steer_phase_sec: float = 0.35
    steer_cooldown_sec: float = 1.0
    hard_stop_cm: int = 15
    caution_cm: int = 28
    stop_confirm_count: int = 1
    recovery_stop_sec: float = 0.2
    ultrasonic_reverse_sec: float = 1.0
    ultrasonic_turn_min_sec: float = 0.6
    ultrasonic_turn_max_sec: float = 1.2
    ultrasonic_wait_sec: float = 2.0
    vlm_stop_scan_turn_sec: float = 0.6
    vlm_stop_scan_wait_sec: float = 2.0
    path_restore_action_sec: float = 1.0
    path_restore_min_counter_turn_sec: float = 0.1
    path_restore_assess_sec: float = 0.3


class ActionDecisionEngine:
    """Stateful policy engine for simple obstacle escape and VLM path restore."""

    def __init__(self, policy: MotionPolicy) -> None:
        self.policy = policy
        self._last_action = VLMAction.SLOW_DOWN
        self._last_non_stop_action = VLMAction.SLOW_DOWN
        self._pending_stop_count = 0
        self._sustained_stop_started_at: float | None = None
        self._steer_phase_action: VLMAction | None = None
        self._steer_phase_started_at = 0.0
        self._last_steer_action: VLMAction | None = None
        self._steer_cooldown_until = 0.0

        self._ultrasonic_reverse_until = 0.0
        self._ultrasonic_turn_until = 0.0
        self._ultrasonic_wait_until = 0.0
        self._ultrasonic_turn_direction: str | None = None
        self._ultrasonic_turn_sec = 0.0

        self._vlm_scan_turn_until = 0.0
        self._vlm_scan_wait_until = 0.0
        self._vlm_scan_direction: str | None = None
        self._vlm_scan_turn_sec = 0.0

        self._path_restore_action_until = 0.0
        self._path_restore_counter_until = 0.0
        self._path_restore_assess_until = 0.0
        self._path_restore_action: VLMAction | None = None
        self._path_restore_counter_direction: str | None = None
        self._path_restore_source_phase: str | None = None
        self._path_restore_counter_sec = 0.0

    def decide(
        self,
        action: VLMAction | None,
        distance_cm: int | None,
        now_mono: float,
        allow_recovery: bool = True,
        ultrasonic_triggered: bool = False,
    ) -> tuple[DutyTuple, str, VLMAction, str]:
        if ultrasonic_triggered:
            self._reset_steer_phase()
            self._reset_sustained_stop()
            self._pending_stop_count = 0
            self._clear_path_restore()
            self._clear_vlm_stop_scan()
            if not self._ultrasonic_escape_active(now_mono):
                self._start_ultrasonic_escape(now_mono)
            phase = self._get_ultrasonic_escape_phase(now_mono)
            if phase is not None:
                return phase

        if self._ultrasonic_escape_active(now_mono):
            if self._ultrasonic_wait_active(now_mono) and self._is_passable_action(action):
                self._clear_ultrasonic_escape()
            else:
                phase = self._get_ultrasonic_escape_phase(now_mono)
                if phase is not None:
                    return phase
                self._clear_ultrasonic_escape()

        path_restore_followup = self._continue_path_restore(
            action=action,
            now_mono=now_mono,
        )
        if path_restore_followup is not None:
            return path_restore_followup

        path_restore_start = self._start_path_restore_if_passable(
            action=action,
            now_mono=now_mono,
            allow_recovery=allow_recovery,
        )
        if path_restore_start is not None:
            return path_restore_start

        vlm_scan_phase = self._continue_vlm_stop_scan(
            action=action,
            now_mono=now_mono,
            allow_recovery=allow_recovery,
        )
        if vlm_scan_phase is not None:
            return vlm_scan_phase

        effective, debounce_detail = self._resolve_action_with_stop_debounce(action)
        effective, steer_cooldown_detail = self._apply_steer_cooldown(
            effective,
            now_mono,
        )

        near_distance = (
            distance_cm is not None and distance_cm <= self.policy.caution_cm
        )

        reason = "vlm_action"
        detail_parts: list[str] = []
        if action is not None:
            detail_parts.append(f"vlm_action={action.value}")
        if debounce_detail:
            detail_parts.append(debounce_detail)
        if steer_cooldown_detail:
            detail_parts.append(steer_cooldown_detail)

        if near_distance and effective == VLMAction.MOVE_FORWARD:
            effective = VLMAction.SLOW_DOWN
            reason = "near_constraint_slow_down"
            detail_parts.append(
                f"distance={distance_cm}cm <= caution_cm={self.policy.caution_cm}cm"
            )

        if effective == VLMAction.STOP:
            stop_duration = self._mark_sustained_stop(now_mono)
            detail_parts.append(
                "sustained_stop_duration="
                f"{stop_duration:.2f}/{self.policy.recovery_stop_sec:.2f}s"
            )
            if allow_recovery and stop_duration >= self.policy.recovery_stop_sec:
                self._start_vlm_stop_scan(now_mono)
                phase = self._get_vlm_stop_scan_phase(now_mono)
                if phase is not None:
                    return phase
        else:
            self._reset_sustained_stop()

        duties, motion_detail = self._duties_for_action(
            effective,
            near_distance,
            now_mono,
        )
        self._remember_effective_action(effective, now_mono)

        if motion_detail:
            detail_parts.append(motion_detail)
        detail_parts.append(f"effective={effective.value}")
        return duties, reason, effective, "; ".join(detail_parts)

    def _start_ultrasonic_escape(self, now_mono: float) -> None:
        self._clear_vlm_stop_scan()
        self._clear_path_restore()
        reverse_sec = max(0.0, self.policy.ultrasonic_reverse_sec)
        turn_min = max(0.0, self.policy.ultrasonic_turn_min_sec)
        turn_max = max(turn_min, self.policy.ultrasonic_turn_max_sec)
        turn_sec = random.uniform(turn_min, turn_max)
        direction = random.choice(("left", "right"))
        wait_sec = max(0.0, self.policy.ultrasonic_wait_sec)

        self._ultrasonic_reverse_until = now_mono + reverse_sec
        self._ultrasonic_turn_until = self._ultrasonic_reverse_until + turn_sec
        self._ultrasonic_wait_until = self._ultrasonic_turn_until + wait_sec
        self._ultrasonic_turn_direction = direction
        self._ultrasonic_turn_sec = turn_sec
        self._last_action = VLMAction.STOP

    def _get_ultrasonic_escape_phase(
        self,
        now_mono: float,
    ) -> tuple[DutyTuple, str, VLMAction, str] | None:
        if not self._ultrasonic_escape_active(now_mono):
            return None

        if now_mono < self._ultrasonic_reverse_until:
            remaining = max(0.0, self._ultrasonic_reverse_until - now_mono)
            speed = max(0, min(4095, self.policy.slow_speed))
            detail = (
                "ultrasonic_phase=reverse "
                f"remaining={remaining:.2f}s "
                f"reverse_sec={self.policy.ultrasonic_reverse_sec:.2f}s"
            )
            return (-speed, -speed, -speed, -speed), "ultrasonic_escape_reverse", VLMAction.STOP, detail

        if now_mono < self._ultrasonic_turn_until:
            direction = self._ultrasonic_turn_direction or "left"
            remaining = max(0.0, self._ultrasonic_turn_until - now_mono)
            detail = (
                "ultrasonic_phase=random_turn "
                f"direction={direction} remaining={remaining:.2f}s "
                f"turn_sec={self._ultrasonic_turn_sec:.2f}s"
            )
            return self._turn_duties(direction), "ultrasonic_escape_random_turn", VLMAction.STOP, detail

        if now_mono < self._ultrasonic_wait_until:
            remaining = max(0.0, self._ultrasonic_wait_until - now_mono)
            detail = (
                "ultrasonic_phase=wait_vlm "
                f"remaining={remaining:.2f}s "
                f"wait_sec={self.policy.ultrasonic_wait_sec:.2f}s"
            )
            return (0, 0, 0, 0), "ultrasonic_escape_wait_vlm", VLMAction.STOP, detail

        return None

    def _continue_vlm_stop_scan(
        self,
        action: VLMAction | None,
        now_mono: float,
        allow_recovery: bool,
    ) -> tuple[DutyTuple, str, VLMAction, str] | None:
        if not self._vlm_stop_scan_active(now_mono):
            if self._vlm_scan_wait_until > 0.0:
                self._clear_vlm_stop_scan()
            return None

        if self._is_passable_action(action):
            return None

        phase = self._get_vlm_stop_scan_phase(now_mono)
        if phase is not None:
            return phase

        self._clear_vlm_stop_scan()
        if allow_recovery and action == VLMAction.STOP:
            self._start_vlm_stop_scan(now_mono)
            return self._get_vlm_stop_scan_phase(now_mono)
        return None

    def _start_vlm_stop_scan(self, now_mono: float) -> None:
        self._clear_ultrasonic_escape()
        self._clear_path_restore()
        turn_sec = max(0.1, self.policy.vlm_stop_scan_turn_sec)
        wait_sec = max(0.0, self.policy.vlm_stop_scan_wait_sec)
        direction = random.choice(("left", "right"))
        self._vlm_scan_turn_until = now_mono + turn_sec
        self._vlm_scan_wait_until = self._vlm_scan_turn_until + wait_sec
        self._vlm_scan_direction = direction
        self._vlm_scan_turn_sec = turn_sec
        self._reset_sustained_stop()
        self._pending_stop_count = 0
        self._last_action = VLMAction.STOP

    def _get_vlm_stop_scan_phase(
        self,
        now_mono: float,
    ) -> tuple[DutyTuple, str, VLMAction, str] | None:
        if not self._vlm_stop_scan_active(now_mono):
            return None

        direction = self._vlm_scan_direction or "left"
        if now_mono < self._vlm_scan_turn_until:
            remaining = max(0.0, self._vlm_scan_turn_until - now_mono)
            detail = (
                "vlm_stop_scan_phase=turn "
                f"direction={direction} remaining={remaining:.2f}s "
                f"turn_sec={self._vlm_scan_turn_sec:.2f}s"
            )
            return self._turn_duties(direction), "vlm_stop_scan_turn", VLMAction.STOP, detail

        if now_mono < self._vlm_scan_wait_until:
            remaining = max(0.0, self._vlm_scan_wait_until - now_mono)
            detail = (
                "vlm_stop_scan_phase=wait_passable "
                f"direction={direction} remaining={remaining:.2f}s "
                f"wait_sec={self.policy.vlm_stop_scan_wait_sec:.2f}s"
            )
            return (0, 0, 0, 0), "vlm_stop_scan_wait", VLMAction.STOP, detail

        return None

    def _resolve_action_with_stop_debounce(
        self,
        action: VLMAction | None,
    ) -> tuple[VLMAction, str | None]:
        if action is None:
            return self._last_action, "no_new_vlm_action_hold_last"

        if action == VLMAction.STOP:
            self._pending_stop_count += 1
            if self._pending_stop_count < self.policy.stop_confirm_count:
                detail = (
                    "stop_debounce_ignore "
                    f"{self._pending_stop_count}/{self.policy.stop_confirm_count}"
                )
                return self._last_non_stop_action, detail
            detail = (
                "stop_debounce_confirmed "
                f"{self._pending_stop_count}/{self.policy.stop_confirm_count}"
            )
            return VLMAction.STOP, detail

        reset_detail = None
        if self._pending_stop_count > 0:
            reset_detail = "stop_debounce_reset"
        self._pending_stop_count = 0
        return action, reset_detail

    def _apply_steer_cooldown(
        self,
        action: VLMAction,
        now_mono: float,
    ) -> tuple[VLMAction, str | None]:
        if action not in (VLMAction.STEER_LEFT, VLMAction.STEER_RIGHT):
            return action, None

        cooldown_sec = max(0.0, self.policy.steer_cooldown_sec)
        if cooldown_sec <= 0.0:
            return action, None

        if action != self._last_steer_action:
            return action, None

        if now_mono >= self._steer_cooldown_until:
            return action, None

        remaining = max(0.0, self._steer_cooldown_until - now_mono)
        self._reset_steer_phase()
        return VLMAction.MOVE_FORWARD, (
            "steer_cooldown_active "
            f"repeat={action.value} remaining={remaining:.2f}s "
            "fallback=Move Forward"
        )

    def _is_passable_action(self, action: VLMAction | None) -> bool:
        return action is not None and action != VLMAction.STOP

    def _start_path_restore_if_passable(
        self,
        action: VLMAction | None,
        now_mono: float,
        allow_recovery: bool,
    ) -> tuple[DutyTuple, str, VLMAction, str] | None:
        if not allow_recovery or not self._is_passable_action(action):
            return None

        context = self._active_path_restore_context(now_mono)
        if context is None or action is None:
            return None

        phase, direction, turn_elapsed, source = context
        self._start_path_restore(
            action=action,
            now_mono=now_mono,
            turn_direction=direction,
            turn_elapsed=turn_elapsed,
            source_phase=phase,
        )
        duties, motion_detail = self._duties_for_action(
            action,
            near_distance=False,
            now_mono=now_mono,
        )
        self._remember_effective_action(action, now_mono)
        detail_parts = [
            f"path_restore_start source={source} phase={phase}",
            f"detected_action={action.value}",
            f"turn_direction={direction}",
            f"counter_direction={self._path_restore_counter_direction}",
            f"action_sec={self.policy.path_restore_action_sec:.2f}",
            f"counter_sec={self._path_restore_counter_sec:.2f}",
        ]
        if motion_detail:
            detail_parts.append(motion_detail)
        return duties, "path_restore_action", action, "; ".join(detail_parts)

    def _continue_path_restore(
        self,
        action: VLMAction | None,
        now_mono: float,
    ) -> tuple[DutyTuple, str, VLMAction, str] | None:
        if not self._path_restore_active():
            return None

        active_phase = self._get_path_restore_phase(now_mono)
        if active_phase is not None:
            return active_phase

        source_phase = self._path_restore_source_phase or "unknown"
        if self._is_passable_action(action) and action is not None:
            self._clear_vlm_stop_scan()
            self._clear_path_restore()
            self._reset_sustained_stop()
            self._pending_stop_count = 0

            duties, motion_detail = self._duties_for_action(
                action,
                near_distance=False,
                now_mono=now_mono,
            )
            self._remember_effective_action(action, now_mono)
            detail_parts = [
                f"path_restore_confirmed source_phase={source_phase}",
                f"next_action={action.value}",
            ]
            if motion_detail:
                detail_parts.append(motion_detail)
            return (
                duties,
                "path_restore_passable_confirmed",
                action,
                "; ".join(detail_parts),
            )

        self._clear_path_restore()
        self._last_action = VLMAction.STOP
        return None

    def _start_path_restore(
        self,
        *,
        action: VLMAction,
        now_mono: float,
        turn_direction: str,
        turn_elapsed: float,
        source_phase: str,
    ) -> None:
        action_sec = max(0.0, self.policy.path_restore_action_sec)
        min_counter_sec = max(0.0, self.policy.path_restore_min_counter_turn_sec)
        counter_sec = max(min_counter_sec, turn_elapsed)
        counter_sec = min(max(0.0, counter_sec), self._max_counter_turn_sec(source_phase))
        assess_sec = max(0.0, self.policy.path_restore_assess_sec)

        self._path_restore_action = action
        self._path_restore_counter_direction = (
            "right" if turn_direction == "left" else "left"
        )
        self._path_restore_source_phase = source_phase
        self._path_restore_counter_sec = counter_sec
        self._path_restore_action_until = now_mono + action_sec
        self._path_restore_counter_until = self._path_restore_action_until + counter_sec
        self._path_restore_assess_until = self._path_restore_counter_until + assess_sec

    def _get_path_restore_phase(
        self,
        now_mono: float,
    ) -> tuple[DutyTuple, str, VLMAction, str] | None:
        action = self._path_restore_action
        if action is None:
            return None

        if now_mono < self._path_restore_action_until:
            remaining = max(0.0, self._path_restore_action_until - now_mono)
            duties, motion_detail = self._duties_for_action(
                action,
                near_distance=False,
                now_mono=now_mono,
            )
            detail_parts = [
                f"path_restore_phase=action remaining={remaining:.2f}s",
                f"source_phase={self._path_restore_source_phase}",
                f"action={action.value}",
            ]
            if motion_detail:
                detail_parts.append(motion_detail)
            return duties, "path_restore_action", action, "; ".join(detail_parts)

        if now_mono < self._path_restore_counter_until:
            direction = self._path_restore_counter_direction or "right"
            remaining = max(0.0, self._path_restore_counter_until - now_mono)
            detail = (
                "path_restore_phase=counter_turn "
                f"direction={direction} remaining={remaining:.2f}s "
                f"source_phase={self._path_restore_source_phase}"
            )
            return self._turn_duties(direction), "path_restore_counter_turn", action, detail

        if now_mono < self._path_restore_assess_until:
            remaining = max(0.0, self._path_restore_assess_until - now_mono)
            detail = (
                "path_restore_phase=assess "
                f"remaining={remaining:.2f}s "
                f"source_phase={self._path_restore_source_phase}"
            )
            return (0, 0, 0, 0), "path_restore_assess", action, detail

        return None

    def _active_path_restore_context(
        self,
        now_mono: float,
    ) -> tuple[str, str, float, str] | None:
        if not self._vlm_stop_scan_active(now_mono):
            return None

        direction = self._vlm_scan_direction or "left"
        scan_start = self._vlm_scan_turn_until - self._vlm_scan_turn_sec
        if scan_start <= now_mono < self._vlm_scan_turn_until:
            return (
                "vlm_stop_scan_turn",
                direction,
                max(0.0, now_mono - scan_start),
                "vlm_stop",
            )
        if self._vlm_scan_turn_until <= now_mono < self._vlm_scan_wait_until:
            return (
                "vlm_stop_scan_wait",
                direction,
                self._vlm_scan_turn_sec,
                "vlm_stop",
            )
        return None

    def _max_counter_turn_sec(self, source_phase: str) -> float:
        if source_phase in ("vlm_stop_scan_turn", "vlm_stop_scan_wait"):
            return max(0.1, self._vlm_scan_turn_sec)
        return max(0.1, self.policy.vlm_stop_scan_turn_sec)

    def _ultrasonic_escape_active(self, now_mono: float) -> bool:
        return self._ultrasonic_wait_until > 0.0 and now_mono < self._ultrasonic_wait_until

    def _ultrasonic_wait_active(self, now_mono: float) -> bool:
        return (
            self._ultrasonic_turn_until > 0.0
            and self._ultrasonic_turn_until <= now_mono < self._ultrasonic_wait_until
        )

    def _clear_ultrasonic_escape(self) -> None:
        self._ultrasonic_reverse_until = 0.0
        self._ultrasonic_turn_until = 0.0
        self._ultrasonic_wait_until = 0.0
        self._ultrasonic_turn_direction = None
        self._ultrasonic_turn_sec = 0.0

    def _vlm_stop_scan_active(self, now_mono: float) -> bool:
        return self._vlm_scan_wait_until > 0.0 and now_mono < self._vlm_scan_wait_until

    def _clear_vlm_stop_scan(self) -> None:
        self._vlm_scan_turn_until = 0.0
        self._vlm_scan_wait_until = 0.0
        self._vlm_scan_direction = None
        self._vlm_scan_turn_sec = 0.0

    def _path_restore_active(self) -> bool:
        return self._path_restore_action is not None

    def _clear_path_restore(self) -> None:
        self._path_restore_action_until = 0.0
        self._path_restore_counter_until = 0.0
        self._path_restore_assess_until = 0.0
        self._path_restore_action = None
        self._path_restore_counter_direction = None
        self._path_restore_source_phase = None
        self._path_restore_counter_sec = 0.0

    def _remember_effective_action(self, action: VLMAction, now_mono: float) -> None:
        self._last_action = action
        if action != VLMAction.STOP:
            self._last_non_stop_action = action
        if action in (VLMAction.STEER_LEFT, VLMAction.STEER_RIGHT):
            self._last_steer_action = action
            self._steer_cooldown_until = (
                now_mono + max(0.0, self.policy.steer_cooldown_sec)
            )

    def _turn_duties(self, direction: str) -> DutyTuple:
        turn_speed = max(0, min(4095, self.policy.turn_speed))
        if direction == "left":
            return (-turn_speed, -turn_speed, turn_speed, turn_speed)
        return (turn_speed, turn_speed, -turn_speed, -turn_speed)

    def _mark_sustained_stop(self, now_mono: float) -> float:
        if self._sustained_stop_started_at is None:
            self._sustained_stop_started_at = now_mono
            return 0.0
        return max(0.0, now_mono - self._sustained_stop_started_at)

    def _reset_sustained_stop(self) -> None:
        self._sustained_stop_started_at = None

    def _reset_steer_phase(self) -> None:
        self._steer_phase_action = None
        self._steer_phase_started_at = 0.0

    def _duties_for_action(
        self,
        action: VLMAction,
        near_distance: bool,
        now_mono: float,
    ) -> tuple[DutyTuple, str | None]:
        slow_speed = max(0, min(4095, self.policy.slow_speed))
        base_speed = max(0, min(4095, self.policy.base_speed))
        turn_speed = max(0, min(4095, self.policy.turn_speed))

        if near_distance:
            base_speed = min(base_speed, slow_speed)

        if action == VLMAction.MOVE_FORWARD:
            self._reset_steer_phase()
            return (base_speed, base_speed, base_speed, base_speed), None
        if action == VLMAction.SLOW_DOWN:
            self._reset_steer_phase()
            return (slow_speed, slow_speed, slow_speed, slow_speed), None
        if action == VLMAction.STEER_LEFT:
            return self._steer_adjust_duties(action, base_speed, turn_speed, now_mono)
        if action == VLMAction.STEER_RIGHT:
            return self._steer_adjust_duties(action, base_speed, turn_speed, now_mono)

        self._reset_steer_phase()
        return (0, 0, 0, 0), None

    def _steer_adjust_duties(
        self,
        action: VLMAction,
        base_speed: int,
        turn_speed: int,
        now_mono: float,
    ) -> tuple[DutyTuple, str]:
        phase_sec = max(0.05, float(self.policy.steer_phase_sec))
        if self._steer_phase_action != action:
            self._steer_phase_action = action
            self._steer_phase_started_at = now_mono

        elapsed = max(0.0, now_mono - self._steer_phase_started_at)
        cycle_sec = 2.0 * phase_sec
        phase_1 = (elapsed % cycle_sec) < phase_sec

        forward_speed = max(base_speed, int(turn_speed * 0.65))
        forward_speed = max(0, min(4095, forward_speed))
        major_delta = max(40, int(turn_speed * 0.35))
        minor_delta = max(25, int(turn_speed * 0.2))

        def _clamp(speed: int) -> int:
            return max(0, min(4095, speed))

        if action == VLMAction.STEER_LEFT:
            if phase_1:
                left_speed = _clamp(forward_speed - major_delta)
                right_speed = _clamp(forward_speed + major_delta)
                phase_name = "phase1_left_bias"
            else:
                left_speed = _clamp(forward_speed + minor_delta)
                right_speed = _clamp(forward_speed - minor_delta)
                phase_name = "phase2_heading_recover_right"
        else:
            if phase_1:
                left_speed = _clamp(forward_speed + major_delta)
                right_speed = _clamp(forward_speed - major_delta)
                phase_name = "phase1_right_bias"
            else:
                left_speed = _clamp(forward_speed - minor_delta)
                right_speed = _clamp(forward_speed + minor_delta)
                phase_name = "phase2_heading_recover_left"

        duties = (left_speed, left_speed, right_speed, right_speed)
        detail = f"steer_curve={phase_name} phase_sec={phase_sec:.2f} heading~stable"
        return duties, detail


class VLMMotionController:
    """Runs control loop without blocking on VLM API latency."""

    def __init__(
        self,
        *,
        action_source: VLMActionSource,
        decision_engine: ActionDecisionEngine,
        motor_setter: Callable[[int, int, int, int], None],
        distance_reader: Optional[Callable[[], int | None]] = None,
        loop_interval_sec: float = 0.1,
        stale_action_timeout_sec: float = 1.0,
    ) -> None:
        self._action_source = action_source
        self._decision_engine = decision_engine
        self._motor_setter = motor_setter
        self._distance_reader = distance_reader
        self._obstacle_source = UltrasonicObstacleSource(
            distance_reader=distance_reader,
            obstacle_trigger_cm=decision_engine.policy.hard_stop_cm,
            caution_cm=decision_engine.policy.caution_cm,
            poll_interval_sec=loop_interval_sec,
        )
        self._loop_interval_sec = max(0.05, loop_interval_sec)
        self._stale_action_timeout_sec = max(0.1, stale_action_timeout_sec)

    def run_until_interrupt(self) -> None:
        self._action_source.start()
        self._obstacle_source.start()
        print("VLM control loop started. Press Ctrl+C to stop.")

        last_duties: DutyTuple | None = None
        last_rule: str | None = None
        last_effective: VLMAction | None = None
        last_raw_action: VLMAction | None = None
        last_source_err: str | None = None
        last_obstacle_err: str | None = None
        last_report_at = 0.0

        try:
            while True:
                now = time.monotonic()
                raw_action, action_age, source_err = self._action_source.latest()
                obstacle = self._obstacle_source.latest()
                action = raw_action
                distance_cm = obstacle.distance_cm
                stale_action = (
                    action_age is None or action_age > self._stale_action_timeout_sec
                )
                allow_recovery = True
                arbitration_source = "vlm"
                if obstacle.obstacle_triggered:
                    action = VLMAction.STOP
                    arbitration_source = "ultrasonic"
                elif stale_action:
                    action = VLMAction.STOP
                    allow_recovery = False
                    arbitration_source = "vlm_stale"

                duties, reason, effective_action, detail = self._decision_engine.decide(
                    action=action,
                    distance_cm=distance_cm,
                    now_mono=now,
                    allow_recovery=allow_recovery,
                    ultrasonic_triggered=obstacle.obstacle_triggered,
                )
                if stale_action and not obstacle.obstacle_triggered:
                    stale_detail = (
                        "vlm_stale_timeout="
                        f"{self._stale_action_timeout_sec:.2f}s "
                        f"action_age={'n/a' if action_age is None else f'{action_age:.2f}s'}"
                    )
                    reason = "vlm_stale_failsafe_stop"
                    detail = stale_detail if not detail else f"{detail}; {stale_detail}"

                if duties != last_duties:
                    self._motor_setter(*duties)
                state_changed = (
                    duties != last_duties
                    or reason != last_rule
                    or effective_action != last_effective
                    or action != last_raw_action
                )
                if state_changed:
                    print(
                        "[EVENT]",
                        f"source={arbitration_source}",
                        f"rule={reason}",
                        f"raw_vlm={raw_action.value if raw_action else 'none'}",
                        f"effective={effective_action.value}",
                        f"state={_duties_to_label(duties)}",
                        f"duties={duties}",
                        f"distance_cm={distance_cm}",
                        f"ultrasonic_triggered={obstacle.obstacle_triggered}",
                        f"detail={detail or 'n/a'}",
                    )

                if source_err != last_source_err:
                    print(
                        "[SOURCE]",
                        "status_api_error=",
                        source_err if source_err else "none",
                    )

                if obstacle.error != last_obstacle_err:
                    print(
                        "[ULTRASONIC]",
                        "distance_error=",
                        obstacle.error if obstacle.error else "none",
                    )

                last_duties = duties
                last_rule = reason
                last_effective = effective_action
                last_raw_action = action
                last_source_err = source_err
                last_obstacle_err = obstacle.error

                if now - last_report_at >= 1.0:
                    age_text = "n/a" if action_age is None else f"{action_age:.2f}s"
                    print(
                        "[HEARTBEAT]",
                        f"source={arbitration_source}",
                        f"raw_vlm={raw_action.value if raw_action else 'none'}",
                        f"effective={effective_action.value}",
                        f"rule={reason}",
                        f"state={_duties_to_label(duties)}",
                        f"distance_cm={distance_cm}",
                        f"ultrasonic_triggered={obstacle.obstacle_triggered}",
                        f"action_age={age_text}",
                        f"source_err={source_err}",
                    )
                    last_report_at = now

                time.sleep(self._loop_interval_sec)
        finally:
            self._motor_setter(0, 0, 0, 0)
            self._obstacle_source.stop()
            self._action_source.stop()
