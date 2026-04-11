#!/usr/bin/env python3
"""
Command-line motor controller for the 4-wheel drive board.

Example usage:
python robot_control/cli.py interactive --speed 1000 --stop-cm 15 --caution-cm 25

"""

import argparse
import json
import random
import select
import sys
import termios
import time
import tty
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gpiozero import DistanceSensor
from robot_inference.robot_control.script.Motor import PWM
from robot_inference.robot_control.script.servo import Servo
from robot_inference.robot_control.script.vlm_action_controller import (
    ActionDecisionEngine,
    MotionPolicy,
    VLMActionSource,
    VLMMotionController,
)


def load_cli_config() -> dict:
    config_path = PROJECT_DIR / "config" / "cli_config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class ObstacleAvoider:
    def __init__(
        self,
        *,
        enabled: bool,
        trigger_pin: int,
        echo_pin: int,
        stop_cm: int,
        caution_cm: int,
        reverse_speed: int,
        turn_speed: int,
        reverse_time: float,
        turn_time: float,
        confirm_pause: float,
        max_duty: int,
        hard_stop_pause: float,
        caution_stop_pause: float,
    ) -> None:
        self.enabled = enabled
        self.stop_cm = stop_cm
        self.caution_cm = caution_cm
        self.reverse_speed = clamp_speed(reverse_speed, max_duty)
        self.turn_speed = clamp_speed(turn_speed, max_duty)
        self.reverse_time = max(0.0, reverse_time)
        self.turn_time = max(0.0, turn_time)
        self.confirm_pause = max(0.0, confirm_pause)
        self.hard_stop_pause = max(0.0, hard_stop_pause)
        self.caution_stop_pause = max(0.0, caution_stop_pause)
        self._last_caution_log = 0.0

        self.sensor = None
        if not enabled:
            return

        if DistanceSensor is None:
            print("Warning: gpiozero is not installed, obstacle avoidance disabled")
            self.enabled = False
            return

        self.sensor = DistanceSensor(echo=echo_pin, trigger=trigger_pin, max_distance=3)

    def distance_cm(self) -> int | None:
        if not self.enabled or self.sensor is None:
            return None

        try:
            return int(self.sensor.distance * 100)
        except Exception:
            return None

    def check_and_avoid(self, duties: tuple[int, int, int, int]) -> bool:
        if not self.enabled or not is_forward_motion(duties):
            return False

        distance = self.distance_cm()
        if distance is None:
            return False

        if distance <= self.stop_cm:
            print(f"Obstacle at {distance} cm: stop, reverse, and reroute")
            stop_motors()
            time.sleep(self.hard_stop_pause)
            self._reverse()
            self._random_turn()
            stop_motors()
            self._confirm_after_turn()
            return True

        if distance <= self.caution_cm:
            now = time.monotonic()
            if now - self._last_caution_log >= 1.0:
                print(f"Obstacle at {distance} cm: stop and turn")
                self._last_caution_log = now
            stop_motors()
            time.sleep(self.caution_stop_pause)
            self._random_turn()
            stop_motors()
            self._confirm_after_turn()
            return True

        return False

    def _reverse(self) -> None:
        speed = -self.reverse_speed
        PWM.setMotorModel(speed, speed, speed, speed)
        time.sleep(self.reverse_time)

    def _random_turn(self) -> None:
        speed = self.turn_speed
        if random.randint(1, 100) <= 50:
            PWM.setMotorModel(-speed, -speed, speed, speed)
            print("Avoidance turn: left")
        else:
            PWM.setMotorModel(speed, speed, -speed, -speed)
            print("Avoidance turn: right")
        time.sleep(self.turn_time)

    def _confirm_after_turn(self) -> None:
        time.sleep(self.confirm_pause)
        distance = self.distance_cm()
        if distance is None:
            print("Post-turn check: distance unavailable")
        elif distance <= self.caution_cm:
            print(f"Post-turn check: obstacle still nearby ({distance} cm)")
        else:
            print(f"Post-turn check: path looks clear ({distance} cm)")


def clamp_speed(value: int, max_duty: int) -> int:
    return max(0, min(max_duty, value))


def clamp_angle(value: int, limits_config: dict) -> int:
    return max(
        limits_config["min_servo_angle"],
        min(limits_config["max_servo_angle"], value),
    )


def set_servo_angle(channel: str, angle: int, limits_config: dict) -> None:
    if Servo is None:
        print("Servo module unavailable")
        return

    servo = Servo()
    target_angle = clamp_angle(angle, limits_config)
    servo.setServoPwm(channel, target_angle)
    print(f"Servo channel {channel} -> {target_angle} deg")


def initialize_servos(config: dict) -> None:
    if Servo is None:
        print("Servo module unavailable, skip servo init")
        return

    limits_config = config["limits"]
    servo_config = config["servo"]
    servo = Servo()
    servo.setServoPwm("0", clamp_angle(servo_config["init_channel_0"], limits_config))
    servo.setServoPwm("1", clamp_angle(servo_config["init_channel_1"], limits_config))
    print(
        f"Servo init: ch0={clamp_angle(servo_config['init_channel_0'], limits_config)} deg, "
        f"ch1={clamp_angle(servo_config['init_channel_1'], limits_config)} deg"
    )


def stop_motors() -> None:
    PWM.setMotorModel(0, 0, 0, 0)
    print("Motors stopped")


def is_forward_motion(duties: tuple[int, int, int, int]) -> bool:
    return all(duty > 0 for duty in duties)


def apply_drive_step(
    duties: tuple[int, int, int, int],
    avoider: ObstacleAvoider | None,
    last_applied: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if avoider and avoider.check_and_avoid(duties):
        return None

    if duties != last_applied:
        PWM.setMotorModel(*duties)
        return duties

    return last_applied


def run_duties(
    d1: int,
    d2: int,
    d3: int,
    d4: int,
    duration: float | None,
    avoider: ObstacleAvoider | None,
) -> None:
    duties = (d1, d2, d3, d4)
    print(f"Applying duties: {d1}, {d2}, {d3}, {d4}")

    if duration is None:
        print("Running continuously. Press Ctrl+C to stop.")
        try:
            last_applied = None
            while True:
                last_applied = apply_drive_step(duties, avoider, last_applied)
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_motors()
        return

    try:
        end_time = time.monotonic() + max(0.0, duration)
        last_applied = None
        while time.monotonic() < end_time:
            last_applied = apply_drive_step(duties, avoider, last_applied)
            time.sleep(0.1)
    finally:
        stop_motors()


def interactive_mode(
    drive_speed: int,
    turn_speed: int,
    avoider: ObstacleAvoider | None,
    max_duty: int,
) -> None:
    drive_speed = clamp_speed(drive_speed, max_duty)
    turn_speed = clamp_speed(turn_speed, max_duty)
    key_to_duties = {
        "i": (drive_speed, drive_speed, drive_speed, drive_speed),
        "k": (-drive_speed, -drive_speed, -drive_speed, -drive_speed),
        "j": (-turn_speed, -turn_speed, turn_speed, turn_speed),
        "l": (turn_speed, turn_speed, -turn_speed, -turn_speed),
    }

    print("Interactive control started")
    print("Use i=forward, j=left, k=backward, l=right")
    print("Use space or s to stop, q to quit")
    print(f"Drive speed={drive_speed}, turn speed={turn_speed}")

    stdin_fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(stdin_fd)

    try:
        tty.setcbreak(stdin_fd)
        stop_motors()
        current_duties = (0, 0, 0, 0)
        last_applied = current_duties
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    break
                if key in (" ", "s"):
                    current_duties = (0, 0, 0, 0)
                    stop_motors()
                    last_applied = current_duties
                    continue

                duties = key_to_duties.get(key)
                if duties is not None:
                    current_duties = duties
                    print(f"Key {key} -> duties {duties}")

            last_applied = apply_drive_step(current_duties, avoider, last_applied)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
        stop_motors()


def add_avoidance_args(parser: argparse.ArgumentParser, config: dict) -> None:
    ultrasonic_config = config["ultrasonic"]
    avoidance_config = config["avoidance"]
    parser.add_argument(
        "--avoid",
        dest="avoid",
        action=argparse.BooleanOptionalAction,
        default=avoidance_config["enabled"],
        help="Enable ultrasonic obstacle avoidance",
    )
    parser.add_argument(
        "--trigger-pin",
        type=int,
        default=ultrasonic_config["trigger_pin"],
        help=(
            "Ultrasonic trigger GPIO pin "
            f"(default: {ultrasonic_config['trigger_pin']})"
        ),
    )
    parser.add_argument(
        "--echo-pin",
        type=int,
        default=ultrasonic_config["echo_pin"],
        help=f"Ultrasonic echo GPIO pin (default: {ultrasonic_config['echo_pin']})",
    )
    parser.add_argument(
        "--stop-cm",
        type=int,
        default=avoidance_config["stop_cm"],
        help=(
            "Distance threshold to stop and reverse "
            f"(default: {avoidance_config['stop_cm']} cm)"
        ),
    )
    parser.add_argument(
        "--caution-cm",
        type=int,
        default=avoidance_config["caution_cm"],
        help=(
            "Distance threshold to stop and turn "
            f"(default: {avoidance_config['caution_cm']} cm)"
        ),
    )
    parser.add_argument(
        "--avoid-turn-speed",
        type=int,
        default=avoidance_config["turn_speed"],
        help=(
            "Turn speed used by obstacle avoidance "
            f"(default: {avoidance_config['turn_speed']})"
        ),
    )


def create_avoider(args: argparse.Namespace, config: dict) -> ObstacleAvoider | None:
    if not hasattr(args, "avoid"):
        return None

    avoidance_config = config["avoidance"]
    max_duty = config["limits"]["max_duty"]
    return ObstacleAvoider(
        enabled=args.avoid,
        trigger_pin=args.trigger_pin,
        echo_pin=args.echo_pin,
        stop_cm=max(1, args.stop_cm),
        caution_cm=max(max(1, args.stop_cm), args.caution_cm),
        reverse_speed=avoidance_config["reverse_speed"],
        turn_speed=clamp_speed(args.avoid_turn_speed, max_duty),
        reverse_time=avoidance_config["reverse_time_sec"],
        turn_time=avoidance_config["turn_time_sec"],
        confirm_pause=avoidance_config["confirm_pause_sec"],
        max_duty=max_duty,
        hard_stop_pause=avoidance_config["hard_stop_pause_sec"],
        caution_stop_pause=avoidance_config["caution_stop_pause_sec"],
    )


def run_vlm_mode(args: argparse.Namespace, config: dict) -> None:
    avoider = create_avoider(args, config)
    distance_reader = avoider.distance_cm if avoider and avoider.enabled else None
    max_duty = config["limits"]["max_duty"]

    action_source = VLMActionSource(
        status_url=args.status_url,
        poll_interval_sec=max(0.05, args.poll_interval),
        timeout_sec=max(0.1, args.api_timeout),
    )
    policy = MotionPolicy(
        base_speed=clamp_speed(args.speed, max_duty),
        slow_speed=clamp_speed(args.slow_speed, max_duty),
        turn_speed=clamp_speed(args.turn_speed, max_duty),
        steer_phase_sec=max(0.05, args.steer_phase_sec),
        steer_cooldown_sec=max(0.0, args.steer_cooldown_sec),
        hard_stop_cm=max(1, args.stop_cm),
        caution_cm=max(max(1, args.stop_cm), args.caution_cm),
        stop_confirm_count=max(1, args.stop_confirm_count),
        recovery_stop_sec=max(0.0, args.recovery_stop_sec),
        ultrasonic_reverse_sec=max(0.0, args.ultrasonic_reverse_sec),
        ultrasonic_turn_min_sec=max(0.0, args.ultrasonic_turn_min_sec),
        ultrasonic_turn_max_sec=max(0.0, args.ultrasonic_turn_max_sec),
        ultrasonic_wait_sec=max(0.0, args.ultrasonic_wait_sec),
        vlm_stop_scan_turn_sec=max(0.1, args.vlm_stop_scan_turn_sec),
        vlm_stop_scan_wait_sec=max(0.0, args.vlm_stop_scan_wait_sec),
        path_restore_action_sec=max(0.0, args.path_restore_action_sec),
        path_restore_min_counter_turn_sec=max(
            0.0, args.path_restore_min_counter_turn_sec
        ),
        path_restore_assess_sec=max(0.0, args.path_restore_assess_sec),
    )
    decision_engine = ActionDecisionEngine(policy=policy)
    controller = VLMMotionController(
        action_source=action_source,
        decision_engine=decision_engine,
        motor_setter=PWM.setMotorModel,
        distance_reader=distance_reader,
        loop_interval_sec=max(0.05, args.control_interval),
        stale_action_timeout_sec=max(0.1, args.vlm_stale_timeout_sec),
    )
    controller.run_until_interrupt()


def build_parser(config: dict) -> argparse.ArgumentParser:
    limits_config = config["limits"]
    manual_config = config["manual_control"]
    vlm_config = config["vlm"]
    parser = argparse.ArgumentParser(description="Motor control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stop_parser = subparsers.add_parser("stop", help="Stop all motors")
    stop_parser.set_defaults(command="stop")

    servo_parser = subparsers.add_parser("servo", help="Set servo angle")
    servo_parser.add_argument("angle", type=int, help="Servo angle (0-180)")
    servo_parser.add_argument(
        "--channel",
        default="0",
        choices=[str(i) for i in range(limits_config["servo_channel_count"])],
        help="Servo channel id (0-7)",
    )
    servo_parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="Hold time in seconds after moving",
    )

    set_parser = subparsers.add_parser("set", help="Set raw wheel duties")
    set_parser.add_argument("duty1", type=int, help="Left upper wheel duty")
    set_parser.add_argument("duty2", type=int, help="Left lower wheel duty")
    set_parser.add_argument("duty3", type=int, help="Right upper wheel duty")
    set_parser.add_argument("duty4", type=int, help="Right lower wheel duty")
    set_parser.add_argument("--duration", type=float, default=None, help="Run time in seconds")
    add_avoidance_args(set_parser, config)

    for name in ("forward", "back", "left", "right"):
        move_parser = subparsers.add_parser(name, help=f"Move {name}")
        move_parser.add_argument(
            "--speed",
            "--drive-speed",
            dest="drive_speed",
            type=int,
            default=manual_config["drive_speed"],
            help=(
                "Drive speed for forward/backward from 0 to "
                f"{limits_config['max_duty']} (default: {manual_config['drive_speed']})"
            ),
        )
        move_parser.add_argument(
            "--turn-speed",
            type=int,
            default=manual_config["turn_speed"],
            help=(
                "Turn speed for left/right from 0 to "
                f"{limits_config['max_duty']} (default: {manual_config['turn_speed']})"
            ),
        )
        move_parser.add_argument("--duration", type=float, default=None, help="Run time in seconds")
        add_avoidance_args(move_parser, config)

    interactive_parser = subparsers.add_parser("interactive", help="Interactive keyboard control")
    interactive_parser.add_argument(
        "--speed",
        "--drive-speed",
        dest="drive_speed",
        type=int,
        default=manual_config["drive_speed"],
        help=(
            "Drive speed for i/k (forward/backward) from 0 to "
            f"{limits_config['max_duty']} (default: {manual_config['drive_speed']})"
        ),
    )
    interactive_parser.add_argument(
        "--turn-speed",
        type=int,
        default=manual_config["turn_speed"],
        help=(
            "Turn speed for j/l (left/right) from 0 to "
            f"{limits_config['max_duty']} (default: {manual_config['turn_speed']})"
        ),
    )
    add_avoidance_args(interactive_parser, config)

    vlm_parser = subparsers.add_parser(
        "vlm",
        help="VLM action based autonomous drive",
    )
    vlm_parser.add_argument(
        "--status-url",
        default=vlm_config["status_url"],
        help=(
            "Inference status endpoint with latest_result JSON "
            f"(default: {vlm_config['status_url']})"
        ),
    )
    vlm_parser.add_argument(
        "--speed",
        type=int,
        default=vlm_config["base_speed"],
        help=(
            "Forward speed from 0 to "
            f"{limits_config['max_duty']} (default: {vlm_config['base_speed']})"
        ),
    )
    vlm_parser.add_argument(
        "--slow-speed",
        type=int,
        default=vlm_config["slow_speed"],
        help=f"Speed used for slow_down action (default: {vlm_config['slow_speed']})",
    )
    vlm_parser.add_argument(
        "--turn-speed",
        type=int,
        default=vlm_config["turn_speed"],
        help=f"Speed used for steer actions (default: {vlm_config['turn_speed']})",
    )
    vlm_parser.add_argument(
        "--steer-phase-sec",
        type=float,
        default=vlm_config["steer_phase_sec"],
        help=(
            "Duration of each steer phase: command direction then opposite direction "
            f"(default: {vlm_config['steer_phase_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--steer-cooldown-sec",
        type=float,
        default=vlm_config["steer_cooldown_sec"],
        help=(
            "Cooldown after a steer action during which repeated same-direction steer "
            "commands are downgraded to forward motion "
            f"(default: {vlm_config['steer_cooldown_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--control-interval",
        type=float,
        default=vlm_config["control_interval_sec"],
        help=(
            "Motor control loop interval in seconds "
            f"(default: {vlm_config['control_interval_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--poll-interval",
        type=float,
        default=vlm_config["poll_interval_sec"],
        help=(
            "Inference status polling interval in seconds "
            f"(default: {vlm_config['poll_interval_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--api-timeout",
        type=float,
        default=vlm_config["api_timeout_sec"],
        help=(
            "Timeout for each status API request in seconds "
            f"(default: {vlm_config['api_timeout_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--stop-confirm-count",
        type=int,
        default=vlm_config["stop_confirm_count"],
        help=(
            "Consecutive stop actions required before honoring stop "
            f"(default: {vlm_config['stop_confirm_count']})"
        ),
    )
    vlm_parser.add_argument(
        "--vlm-stale-timeout-sec",
        type=float,
        default=vlm_config["vlm_stale_timeout_sec"],
        help=(
            "Failsafe timeout before stale VLM input is downgraded to stop "
            f"(default: {vlm_config['vlm_stale_timeout_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--recovery-stop-sec",
        type=float,
        default=vlm_config["recovery_stop_sec"],
        help=(
            "Continuous confirmed VLM stop duration needed to trigger stop scan "
            f"(default: {vlm_config['recovery_stop_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--ultrasonic-reverse-sec",
        type=float,
        default=vlm_config["ultrasonic_reverse_sec"],
        help=(
            "Reverse duration after ultrasonic trigger "
            f"(default: {vlm_config['ultrasonic_reverse_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--ultrasonic-turn-min-sec",
        type=float,
        default=vlm_config["ultrasonic_turn_min_sec"],
        help=(
            "Minimum random turn duration after ultrasonic reverse "
            f"(default: {vlm_config['ultrasonic_turn_min_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--ultrasonic-turn-max-sec",
        type=float,
        default=vlm_config["ultrasonic_turn_max_sec"],
        help=(
            "Maximum random turn duration after ultrasonic reverse "
            f"(default: {vlm_config['ultrasonic_turn_max_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--ultrasonic-wait-sec",
        type=float,
        default=vlm_config["ultrasonic_wait_sec"],
        help=(
            "Stop duration after ultrasonic random turn while waiting for VLM "
            f"(default: {vlm_config['ultrasonic_wait_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--vlm-stop-scan-turn-sec",
        type=float,
        default=vlm_config["vlm_stop_scan_turn_sec"],
        help=(
            "Turn duration used when only VLM keeps reporting stop "
            f"(default: {vlm_config['vlm_stop_scan_turn_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--vlm-stop-scan-wait-sec",
        type=float,
        default=vlm_config["vlm_stop_scan_wait_sec"],
        help=(
            "Stop duration after VLM stop scan turn while waiting for VLM "
            f"(default: {vlm_config['vlm_stop_scan_wait_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--path-restore-action-sec",
        type=float,
        default=vlm_config["path_restore_action_sec"],
        help=(
            "Duration to follow a passable VLM action during recovery before "
            "counter-turning back to the original path "
            f"(default: {vlm_config['path_restore_action_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--path-restore-min-counter-turn-sec",
        type=float,
        default=vlm_config["path_restore_min_counter_turn_sec"],
        help=(
            "Minimum counter-turn duration after path restore action "
            f"(default: {vlm_config['path_restore_min_counter_turn_sec']})"
        ),
    )
    vlm_parser.add_argument(
        "--path-restore-assess-sec",
        type=float,
        default=vlm_config["path_restore_assess_sec"],
        help=(
            "Pause duration after counter-turning to reassess VLM passability "
            f"(default: {vlm_config['path_restore_assess_sec']})"
        ),
    )
    add_avoidance_args(vlm_parser, config)

    return parser


def command_to_duties(args: argparse.Namespace) -> tuple[int, int, int, int]:
    drive_speed = clamp_speed(getattr(args, "drive_speed", 0), args.max_duty)
    turn_speed = clamp_speed(getattr(args, "turn_speed", 0), args.max_duty)

    if args.command == "forward":
        return drive_speed, drive_speed, drive_speed, drive_speed
    if args.command == "back":
        return -drive_speed, -drive_speed, -drive_speed, -drive_speed
    if args.command == "left":
        return -turn_speed, -turn_speed, turn_speed, turn_speed
    if args.command == "right":
        return turn_speed, turn_speed, -turn_speed, -turn_speed

    return args.duty1, args.duty2, args.duty3, args.duty4


def main() -> None:
    config = load_cli_config()
    parser = build_parser(config)
    args = parser.parse_args()
    args.max_duty = config["limits"]["max_duty"]

    initialize_servos(config)
    if args.command == "stop":
        stop_motors()
        return

    if args.command == "servo":
        set_servo_angle(args.channel, args.angle, config["limits"])
        if args.hold > 0:
            time.sleep(args.hold)
        return

    if args.command == "interactive":
        interactive_mode(
            args.drive_speed,
            args.turn_speed,
            create_avoider(args, config),
            config["limits"]["max_duty"],
        )
        return

    if args.command == "vlm":
        run_vlm_mode(args, config)
        return

    duties = command_to_duties(args)
    run_duties(*duties, duration=args.duration, avoider=create_avoider(args, config))


if __name__ == "__main__":
    main()
