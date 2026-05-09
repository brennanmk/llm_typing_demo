#!/usr/bin/env python3
"""
State machine for demo - runs the full LLM typing planning pipeline
and executes the resulting plan.
"""

import argparse
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import SetBool

from llm_typing_demo.action import ScanScene, PlungeGrasp, WipeBoard
from llm_typing_demo.msg import ObjectDetection
from llm_typing_demo.srv import ClearBeliefs, GenerateProblem, GetBeliefs, RemoveBelief

from llm_typing_planner.api import solve


class ReplanRequired(Exception):
    """Raised during plan execution to signal that a full re-plan is needed."""


class ActionAborted(Exception):
    """Raised when the user declines to retry a failed action."""


def parse_plan(plan_strs: list) -> list:
    """Parse planner output into seperate actions.

    :param plan_strs: List of raw action strings from the planner.
    :return: List of (name, args) tuples where args is a list of strings.
    """
    actions = []
    for line in plan_strs:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        m = re.match(r"\((\S+)(.*)\)$", line)
        if m:
            name = m.group(1).lower()
            args = m.group(2).split()
            actions.append((name, args))
    return actions


class PlanExecutorNode(Node):
    """Control node that orchestrates high level behaviour"""

    def __init__(self) -> None:
        super().__init__("plan_executor")
        self.object_poses = {}

        self._scan_client = ActionClient(self, ScanScene, "scan_scene")
        self._gen_client = self.create_client(GenerateProblem, "generate_problem")
        self._grasp_client = ActionClient(self, PlungeGrasp, "stretch_plunge_grasp")
        self._wipe_client = ActionClient(self, WipeBoard, "wipe_board")
        self._remove_belief_client = self.create_client(RemoveBelief, "remove_belief")
        self._get_beliefs_client = self.create_client(GetBeliefs, "get_beliefs")
        self._clear_beliefs_client = self.create_client(ClearBeliefs, "clear_beliefs")
        self._freeze_beliefs_client = self.create_client(SetBool, "freeze_beliefs")
        self._pause_detection_client = self.create_client(SetBool, "pause_detection")

    def _call_service(self, client, request, timeout: float = 10.0):
        """simple wrapper for a blocking service call.

        :param client: ServiceClient to call.
        :param request: Service request message.
        :param timeout: Maximum seconds to wait for a response.
        :return: Service response, or None on timeout.
        """
        client.wait_for_service()
        future = client.call_async(request)
        start = time.time()
        while rclpy.ok() and not future.done() and (time.time() - start) < timeout:
            time.sleep(0.05)
        return future.result() if future.done() else None

    def _set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze both the belief node and the detection timer together.

        :param frozen: True to freeze, False to unfreeze.
        """
        req = SetBool.Request()
        req.data = frozen
        self._call_service(self._freeze_beliefs_client, req)
        self._call_service(self._pause_detection_client, req)

    def scan_and_generate(self, goal: str) -> str:
        """scan then read the belief state to produce a PDDL problem.

        :param goal: PDDL goal expression string, e.g. (whiteboard_clean whiteboard_0).
        :return: Untyped PDDL problem string ready for the LLM typing planner.
        """
        # Unfreeze, clear beliefs, scan, then freeze again
        self._set_frozen(False)
        self._call_service(self._clear_beliefs_client, ClearBeliefs.Request())
        self._trigger_active_scan()
        self._set_frozen(True)

        detected = self._get_current_beliefs()

        self.get_logger().info("Waiting for generate_problem service...")
        req = GenerateProblem.Request()
        req.goal = goal
        req.detected_objects = list(detected)

        resp = self._call_service(self._gen_client, req)
        if resp is None or not resp.success:
            msg = resp.message if resp else "timeout"
            raise RuntimeError(f"generate_problem failed: {msg}")

        self.object_poses = {obj.pddl_name: obj for obj in resp.objects}
        return resp.problem_pddl

    def _rescan_beliefs(self) -> None:
        """Clear beliefs and trigger a fresh scan, keeping the existing PDDL problem string."""
        self._set_frozen(False)
        self._call_service(self._clear_beliefs_client, ClearBeliefs.Request())
        self._trigger_active_scan()
        self._set_frozen(True)

    def _trigger_active_scan(self) -> None:
        """Send a ScanScene goal and block until the head pan completes."""
        self.get_logger().info("Waiting for scan_scene server...")
        self._scan_client.wait_for_server()
        self.get_logger().info("Scanning scene...")

        # create a threading event to wait on
        done = threading.Event()
        failed = [False]

        def result_callback(future):
            done.set()

        def goal_callback(future):
            gh = future.result()
            if not gh.accepted:
                self.get_logger().error("scan_scene goal rejected")
                failed[0] = True
                done.set()
                return
            gh.get_result_async().add_done_callback(result_callback)

        self._scan_client.send_goal_async(ScanScene.Goal()).add_done_callback(
            goal_callback
        )
        done.wait()

        if failed[0]:
            raise RuntimeError("scan_scene failed")

    def _get_current_beliefs(self) -> list:
        """Query the belief node.

        :return: List of ObjectDetection messages from the belief state.
        """
        result = self._call_service(self._get_beliefs_client, GetBeliefs.Request())
        if result is None:
            raise RuntimeError("get_beliefs call failed")
        self.get_logger().info(f"get_beliefs returned {len(result.objects)} objects")
        return result.objects

    def _refresh_pose(self, obj_name: str) -> bool:
        """Update the stored pose of a given object

        :param obj_name: PDDL name of the object (key in ``self.object_poses``).
        :return: True if the pose was refreshed, False if the
                 object is no longer present in the belief state.
        """
        entry = self.object_poses.get(obj_name)
        if entry is None or not entry.belief_id:
            return True

        beliefs = self._get_current_beliefs()
        for b in beliefs:
            if b.belief_id == entry.belief_id:
                entry.center_x = b.center_x
                entry.center_y = b.center_y
                entry.center_z = b.center_z
                entry.size_y = b.size_y
                entry.size_z = b.size_z
                return True

        return False

    def _call_action(self, client, goal, timeout: float = 120.0):
        """another wrapper. send an action goal and block until the result is received.

        :param client: ActionClient to send the goal through.
        :param goal: Action goal message to send.
        :param timeout: Maximum seconds to wait for the result before logging a timeout.
        :return: The wrapped action result, or ``None`` on timeout or rejection.
        """
        done = threading.Event()
        result_ref = [None]

        def result_callback(future):
            result_ref[0] = future.result()
            done.set()

        def goal_callback(future):
            gh = future.result()
            if not gh.accepted:
                self.get_logger().error("Goal was rejected by server")
                done.set()
                return
            gh.get_result_async().add_done_callback(result_callback)

        client.send_goal_async(goal).add_done_callback(goal_callback)
        if not done.wait(timeout=timeout):
            self.get_logger().error("Action timed out")
        return result_ref[0]

    def do_pickup(self, obj_name: str) -> None:
        """Execute a plunge grasp on the named object.

        :param obj_name: PDDL name of the object to grasp.
        """
        if not self._refresh_pose(obj_name):
            raise ReplanRequired(f"{obj_name} not found in belief state before pickup")

        entry = self.object_poses[obj_name]

        goal = PlungeGrasp.Goal()
        goal.position.x = entry.center_x
        goal.position.y = entry.center_y
        goal.position.z = entry.center_z

        self.get_logger().info(
            f"pickup: grasping {obj_name} at map "
            f"({entry.center_x:.2f}, {entry.center_y:.2f}, {entry.center_z:.2f})"
        )
        self._grasp_client.wait_for_server()
        while True:
            self._call_action(self._grasp_client, goal)
            ans = input("Did pickup succeed? [Y/n]: ").strip().lower()
            if ans != "n":
                break
            ans = input("Retry pickup? [Y/n]: ").strip().lower()
            if ans == "n":
                raise ActionAborted(f"pickup {obj_name} declined by user")
            self._rescan_beliefs()
            if not self._refresh_pose(obj_name):
                raise ReplanRequired(f"{obj_name} not found after rescan")
            goal.position.x = entry.center_x
            goal.position.y = entry.center_y
            goal.position.z = entry.center_z
            self.get_logger().info(
                f"pickup retry -> {obj_name} at map "
                f"({entry.center_x:.2f}, {entry.center_y:.2f}, {entry.center_z:.2f})"
            )

        # the grasped item is no longer in our belief system, its in our hand!
        belief_id = entry.belief_id
        if belief_id:
            req = RemoveBelief.Request()
            req.belief_id = belief_id
            result = self._call_service(self._remove_belief_client, req)
            if result is None or not result.success:
                self.get_logger().warn(f"pickup: remove_belief({belief_id}) failed")

    def do_wipe(self, wb_name: str) -> None:
        """Send a wipe_whiteboard goal.

        :param wb_name: PDDL name of the whiteboard.
        """
        if not self._refresh_pose(wb_name):
            raise ReplanRequired(f"{wb_name} not found in belief state before wipe")

        entry = self.object_poses[wb_name]

        det = ObjectDetection()
        det.class_name = entry.class_name
        det.center_x = entry.center_x
        det.center_y = entry.center_y
        det.center_z = entry.center_z
        det.size_y = entry.size_y
        det.size_z = entry.size_z

        goal = WipeBoard.Goal()
        goal.board = det

        self.get_logger().info(
            f"clean_whiteboard -> wiping {wb_name} at map "
            f"({entry.center_x:.2f}, {entry.center_y:.2f}, {entry.center_z:.2f}), "
            f"size {det.size_y:.2f}w x {det.size_z:.2f}h"
        )
        self._wipe_client.wait_for_server()
        while True:
            result = self._call_action(self._wipe_client, goal, timeout=240.0)
            if result is not None and result.result.success:
                return
            self.get_logger().error(f"Wipe {wb_name} failed.")
            ans = input("Retry wipe? [Y/n]: ").strip().lower()
            if ans == "n":
                raise ActionAborted(f"wipe {wb_name} declined by user")
            self._rescan_beliefs()
            if not self._refresh_pose(wb_name):
                raise ReplanRequired(f"{wb_name} not found after rescan")
            det.center_x = entry.center_x
            det.center_y = entry.center_y
            det.center_z = entry.center_z
            det.size_y = entry.size_y
            det.size_z = entry.size_z
            self.get_logger().info(
                f"wipe retry -> {wb_name} at map "
                f"({entry.center_x:.2f}, {entry.center_y:.2f}, {entry.center_z:.2f}), "
                f"size {det.size_y:.2f}w x {det.size_z:.2f}h"
            )

    def run_plan(self, plan: list) -> None:
        """Sequentially dispatch each action in the plan.

        :param plan: List of (action_name, args).
        """
        for action_name, args in plan:
            self.get_logger().info(f"[plan] ({action_name} {' '.join(args)})")

            if action_name == "pickup":
                self.do_pickup(args[1])
            elif action_name == "clean_whiteboard":
                self.do_wipe(args[1])
            else:
                self.get_logger().warn(f"Unknown action '{action_name}' — skipping")

        self.get_logger().info("Plan complete.")

    def execute(
        self,
        goal: str,
        domain_str: str,
        model: str,
        provider: str = "ollama",
        ollama_host: str = "http://127.0.0.1:11434",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_attempts: int = 3,
    ) -> bool:
        """Perceive -> plan -> execute loop.

        :param goal: PDDL goal expression string.
        :param domain_str: PDDL domain string.
        :param model: Model name for the LLM planner.
        :param provider: LLM provider ("ollama" or "openai").
        :param ollama_host: Ollama server URL.
        :param base_url: OpenAI-compatible base URL (e.g. OpenRouter).
        :param api_key: API key for hosted providers.
        :param max_attempts: Maximum number of full perceive→plan→execute cycles.
        :return: True if the goal was achieved, False otherwise.
        """
        for attempt in range(1, max_attempts + 1):
            self.get_logger().info(
                f"Attempt {attempt}/{max_attempts}: scanning scene..."
            )

            # scan
            while True:
                try:
                    problem_str = self.scan_and_generate(goal)
                except RuntimeError as e:
                    self.get_logger().error(f"scan_and_generate failed: {e}")
                    return False

                print("\n=== Detected objects ===")
                for name, obj in self.object_poses.items():
                    if name == "start":
                        continue
                    print(
                        f"  {name} ({obj.class_name}): "
                        f"({obj.center_x:.2f}, {obj.center_y:.2f}, {obj.center_z:.2f})"
                    )

                ans = input("\nRescan? [y/N]: ").strip().lower()
                if ans != "y":
                    break

            # Plan with typing planner
            while True:
                self.get_logger().info("Running LLM typing planner...")
                plan_strs = solve(
                    domain_str,
                    problem_str,
                    model=model,
                    provider=provider,
                    ollama_host=ollama_host,
                    base_url=base_url,
                    api_key=api_key,
                    logger=self.get_logger(),
                )

                if plan_strs is not None:
                    break

                self.get_logger().error("Planning failed.")
                ans = input("Retry planning? [Y/n]: ").strip().lower()
                if ans == "n":
                    return False

            # Execute
            print(f"\n=== Plan ({len(plan_strs)} steps) ===")
            for i, step in enumerate(plan_strs, 1):
                print(f"  {i:2d}. {step}")

            ans = input("\nExecute plan? [Y/n]: ").strip().lower()
            if ans == "n":
                self.get_logger().info("Execution cancelled by user.")
                return False

            try:
                self.run_plan(parse_plan(plan_strs))
                return True
            except ActionAborted as e:
                self.get_logger().info(f"Action aborted: {e}. Returning to home pose.")
                return False
            except ReplanRequired as e:
                self.get_logger().warn(f"Replanning required: {e}")
                if attempt < max_attempts:
                    self.get_logger().info("Re-scanning and re-planning...")
                    continue
                self.get_logger().error("Max replan attempts reached.")
                return False

        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _default_domain = (
        Path(get_package_share_directory("llm_typing_demo"))
        / "config"
        / "stretch_domain.pddl"
    )
    parser.add_argument(
        "--domain",
        default=str(_default_domain),
        help="Path to domain PDDL (default: package config/stretch_domain.pddl)",
    )
    parser.add_argument(
        "--goal",
        default="(whiteboard_clean whiteboard_0)",
        help='PDDL goal expression (default: "(whiteboard_clean whiteboard_0)")',
    )
    parser.add_argument("--model", default="qwen/qwen3-32b", help="Model name")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["ollama", "openai"],
        help="LLM provider",
    )
    parser.add_argument(
        "--ollama-host", default="http://127.0.0.1:11434", help="Ollama server URL"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.groq.com/openai/v1",
        help="OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum perceive→plan→execute attempts before giving up (default: 3)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    with open(args.domain) as f:
        domain_str = f.read()

    rclpy.init()
    node = PlanExecutorNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(1.0)  # let TF buffer start populating

    success = node.execute(
        args.goal,
        domain_str,
        args.model,
        provider=args.provider,
        ollama_host=args.ollama_host,
        base_url=args.base_url,
        api_key=args.api_key,
        max_attempts=args.max_attempts,
    )

    rclpy.shutdown()
    sys.exit(0 if success else 1)
