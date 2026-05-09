# ROS Course Final Project - LLM Typing Planner

## Overview
This project consists of an implementation to demonstrate and
experiment with using an LLM to type PDDL objects at runtime.

This package is designed to work with the `stretch-re1` robot.

The demo consists of:
* Sweeping the camera and classifying + segmenting all observed items with DINO and SAM
* Generating a PDDL problem with generic types for the observed items
* Solving the task with the LLM Typing Planner
* Executing the generated plan

Specifically, the task this demo is designed to do is pick up an
eraser like item, and wipe off a surface.

## Setup
- NOTE: While this project does not strictly require a GPU, one is highly recommended to run inference for DINO and SAM.
- Ensure that `control_msgs` and `image_geometry` are installed with `sudo apt install ros-humble-control-msgs ros-humble-image-geometry` 
- Make sure that the ROS package is cloned in a ROS workspace on all systems (the robot + an external machine, as the stretch's onboard computer is to weak for inference). Build with `colcon build` and source `install/setup.bash`
- Ensure that all of the packages inside `llm_typing_demo/requirements.txt` are installed, this can be done with `pip install -r requirements.txt`
- SAM and DINO are pulled from huggingface, and so it might be beneficial to set a `HF_TOKEN` for faster downloads
- By default the plan executor is designed to use https://groq.com/ for inference. This can be overriden with any openai v1 style endpoint (using the plan executors `--base-url` argument). Make sure that an `OPENAI_API_KEY` is set as an environment variable before running, as this will be used for inference.

## External Tools
This project has a git submodule for the LLM Typing Planner
implementation. This project has its own set of install instructions,
and relies on tools including langchain and pyperplan. All such
dependencies are available in the LLM Typing Planners `requirements.txt`

## Launching Demo
- Make sure that all computers are on a shared middleware (CycloneDDS is recommended), and have matching DOMAIN_ID's 
- On the streth re1's computer run `ros2 launch llm_typing_demo stretch_bringup.launch.py` to bring the robot up
- On another machine (preferably with a GPU) run `ros2 launch llm_typing_demo stretch_demo.launch.py` to bring the robot up
- To run the demo executor (on either machine) run `ros2 run llm_typing_demo plan_executor.py` 

## Submissions 
- Project System Architecture Final Draft & Preliminary Implementation: https://github.com/brennanmk/llm_typing_demo/commit/6b696dc94ff4e3edf7348a2a51f574622dd5fe14
- Incorporate MoveIt! & Nav2 into Project: https://github.com/brennanmk/llm_typing_demo/commit/47ef5e75028a7ebb0b764334a24c9c900ba59b1f
- Incorporate Perception into Project: https://github.com/brennanmk/llm_typing_demo/commit/b88e9964fed8edf6cc28cf39b5ffa8146a7dc967
-  Incorporate Supervisory Control into Project: https://github.com/brennanmk/llm_typing_demo/commit/336ccae85934bf4ad2a5410037c03cf24ffc71e3 
-  Add advanced custom component to project: https://github.com/brennanmk/llm_typing_demo/commit/995496acc03eb6cff3fe3198dd5cfa238e78553e 
- Final submission: latest commit

## Project Components
### Nav2
Can be found in https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/navigate_to.py

rtab is launched on the stretch to make a map. This map is also used in https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/object_detection.py to anchor detected objects to a fixed frame (the map).

### MoveIt!
In addition to the stretch plunge grasp, a plunge grasp was implemented in MoveIt for the Kinova Gen3 lite https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/moveit_plunge_grasp.py 

For the actual demo, MoveIt! was not used as the stretch does not support it.

### Supervisory Control
The plan executor node acts as the high level supervisory controller for the demo https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/scripts/plan_executor.py

### Perception
The object detection node - https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/object_detection.py - uses the realsense camera on the stretch along with DINO and SAM to detect objects, and get object transforms relative to the robot.

### Custom Component
The most interesting custom component (aside from the typing planner itself) is probably the belief node - https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/belief_node.py . This node is used to track objects in the scene.

Additionally, the problem generator takes detected objects and turns them into PDDL to be solved by the llm typing planner. https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/problem_generator.py

### Written by hand
The scan scene node was written by hand: https://github.com/brennanmk/llm_typing_demo/blob/main/llm_typing_demo/src/scan_scene.py

## Demo
A demo video can be found at https://drive.google.com/file/d/19W0J5Q4Kj6AV3E5iSEv_xP7e-SUCmyiK/view?usp=drive_link

## ROS Support
This assignment was tested with ROS Humble, although it may compile to other versions like Kilted.
