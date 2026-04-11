# Robot Motion Algorithm

## Overview

The VLM drive controller now uses two independent monitoring channels:

- VLM action output from `/api/status`
- ultrasonic obstacle trigger state

The ultrasonic channel has priority. When it triggers, VLM actions are ignored until the short escape sequence finishes or the robot reaches the VLM-wait phase.

## Main Loop

Each control tick does this:

1. Read latest ultrasonic state.
2. Read latest VLM action.
3. If ultrasonic is triggered, run ultrasonic escape.
4. If ultrasonic is clear, use VLM action.
5. If VLM input is stale, fail safe to `Stop` without starting recovery.
6. Convert the final state into four wheel PWM duties.

## Ultrasonic Escape

When `distance_cm <= hard_stop_cm`, the controller runs a simple escape sequence:

1. Reverse for `ultrasonic_reverse_sec`.
2. Randomly choose left or right.
3. Turn in that random direction for a random duration between `ultrasonic_turn_min_sec` and `ultrasonic_turn_max_sec`.
4. Stop for `ultrasonic_wait_sec` and wait for a VLM action.

Default timing:

- reverse: `1.0s`
- random turn: `0.6s` to `1.2s`
- wait after turn: `2.0s`

If a passable VLM action appears during the wait phase, the ultrasonic escape state clears and normal VLM control resumes. If ultrasonic triggers again, a fresh escape sequence starts.

## VLM Stop Scan

When ultrasonic is clear and VLM keeps reporting confirmed `Stop` for at least `recovery_stop_sec`, the controller does not run the old multi-step probe/recovery/final-fallback sequence anymore.

Instead it runs a small scan:

1. Randomly choose left or right.
2. Turn for `vlm_stop_scan_turn_sec`.
3. Stop for `vlm_stop_scan_wait_sec` and wait for VLM to report a passable action.

Default timing:

- stop hold before scan: `1.0s`
- scan turn: `0.6s`
- wait after scan: `2.0s`

If VLM still reports `Stop` after the wait, the controller can start another simple scan. There is no reverse, no left-pause-right-pause probe, no formal recovery chain, and no final fallback turn.

## Path Restore

Path restore only applies to the VLM-only stop scan, not to ultrasonic escape.

If ultrasonic is clear and a passable VLM action appears while the robot is turning or waiting after a VLM stop scan:

1. Execute that VLM action for `path_restore_action_sec`.
2. Counter-turn toward the original heading.
3. Pause for `path_restore_assess_sec`.
4. If VLM is still passable, clear scan state and resume normal VLM control.
5. If VLM returns to `Stop`, continue with the simple VLM stop scan behavior.

Counter-turn direction:

- If the scan was left, counter-turn right.
- If the scan was right, counter-turn left.

The counter-turn duration is based on how long the scan turn had already run, with a minimum of `path_restore_min_counter_turn_sec`.

## Normal VLM Actions

VLM actions map to duties as before:

- `Move Forward`: all wheels positive at `base_speed`
- `Slow Down`: all wheels positive at `slow_speed`
- `Stop`: all wheels zero, with stop debounce
- `Steer Left`: curved left steering
- `Steer Right`: curved right steering

Repeated same-direction steering is still cooled down by `steer_cooldown_sec`.

## Key Parameters

These values live under `vlm` in `robot_control/config/cli_config.json`.

- `base_speed`: normal forward speed
- `slow_speed`: cautious forward and reverse escape speed
- `turn_speed`: turn speed
- `steer_phase_sec`: steering curve phase duration
- `steer_cooldown_sec`: repeated same-direction steering cooldown
- `ultrasonic_poll_interval_sec`: ultrasonic channel polling interval
- `vlm_stale_timeout_sec`: stale VLM cutoff
- `stop_confirm_count`: stop debounce threshold
- `recovery_stop_sec`: confirmed VLM stop duration before simple scan
- `ultrasonic_reverse_sec`: reverse duration after ultrasonic trigger
- `ultrasonic_turn_min_sec`: minimum random ultrasonic turn duration
- `ultrasonic_turn_max_sec`: maximum random ultrasonic turn duration
- `ultrasonic_wait_sec`: stop/wait duration after ultrasonic random turn
- `vlm_stop_scan_turn_sec`: simple scan turn duration for VLM-only stop
- `vlm_stop_scan_wait_sec`: stop/wait duration after VLM-only scan turn
- `path_restore_action_sec`: passable action duration before counter-turning
- `path_restore_min_counter_turn_sec`: minimum counter-turn duration
- `path_restore_assess_sec`: pause after counter-turn before reassessing VLM
- `hard_stop_cm`: ultrasonic obstacle trigger threshold
- `caution_cm`: slow-down threshold
