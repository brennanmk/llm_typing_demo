#!/usr/bin/env python3
"""
node the implements problem generator service. returns a string
containing a PDDL problem.

Given a goal and a list of detected objects, formats and returns a
PDDL problem string.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from llm_typing_demo.msg import ObjectPose
from llm_typing_demo.srv import GenerateProblem

# Classes that map wipable objects to whiteboard, useful for supporting odd dino types
WHITEBOARD_CLASSES = {
    "whiteboard",
    "blackboard",
    "chalkboard",
    "black square",
    "black_square",
}

# generic template, we fill in the objects
PROBLEM_TEMPLATE = """\
(define (problem stretch_cleanup_0)
  (:domain stretch_cleanup)

  (:objects
{objects}
  )

  (:init
    (hand_empty stretch)
  )

  (:goal {goal})
)
"""


def _build_object_poses(detected_objects: list) -> list:
    """Assign PDDL names in the form of {class}_{index} to detected objects.

    We also strip spaces to make we end up with valid PDDL

    :param detected_objects: List of ObjectDetection messages from the scene scan.
    :return: List of ObjectPose messages with pddl_name and pose fields populated.
    """
    counts = {}
    result = []
    for obj in detected_objects:
        cls = (
            "whiteboard"
            if obj.class_name.lower() in WHITEBOARD_CLASSES
            else obj.class_name.replace(" ", "_").lower()
        )
        idx = counts.get(cls, 0)
        counts[cls] = idx + 1

        pose = ObjectPose()
        pose.belief_id = obj.belief_id
        pose.pddl_name = f"{cls}_{idx}"
        pose.class_name = obj.class_name
        pose.center_x = obj.center_x
        pose.center_y = obj.center_y
        pose.center_z = obj.center_z
        pose.size_y = obj.size_y
        pose.size_z = obj.size_z
        result.append(pose)

    return result


class ProblemGeneratorNode(Node):
    """Service node that formats an untyped PDDL problem from a list of detected objects."""

    def __init__(self) -> None:
        super().__init__("problem_generator")
        self.create_service(GenerateProblem, "generate_problem", self._handle_generate)
        self.get_logger().info("problem_generator ready.")

    def _handle_generate(
        self,
        request: GenerateProblem.Request,
        response: GenerateProblem.Response,
    ) -> GenerateProblem.Response:
        """Handle a generate_problem service request.

        Assigns PDDL names to all detected objects, formats the PDDL problem string,
        and returns both along with the ObjectPose list for the caller's pose tracking.

        :param request: GenerateProblem.Request with goal string and detected objects.
        :param response: GenerateProblem.Response to populate and return.
        :return: Populated GenerateProblem.Response.
        """
        objects = _build_object_poses(request.detected_objects)
        lines = ["    stretch - robot"]
        for o in objects:
            pddl_type = (
                "whiteboard"
                if o.class_name.lower() in WHITEBOARD_CLASSES
                else "graspable"
            )
            lines.append(f"    {o.pddl_name} - {pddl_type}")
        objects_block = "\n".join(lines)

        response.success = True
        response.message = "ok"
        response.problem_pddl = PROBLEM_TEMPLATE.format(
            objects=objects_block, goal=request.goal
        )
        response.objects = objects
        self.get_logger().info(
            f"Generated problem with {len(objects)} objects: "
            f"{[o.pddl_name for o in objects]}"
        )
        return response


if __name__ == "__main__":
    rclpy.init()
    node = ProblemGeneratorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
