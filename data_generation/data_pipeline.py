"""
AirNav Data Generation Pipeline

Four-Step Pipeline for automatic VLN data generation:
    Step 1: Start & Target Selection
        - Random start point sampling
        - Target object selection
        - MLLM target description generation
        - Quality filtering
    Step 2: Landmark Planning
        - Landmark identification
        - Distance constraints
        - Semantic refinement
    Step 3: Trajectory Synthesis
        - Look-Ahead strategy
        - Trajectory concatenation
    Step 4: Instruction Generation
        - Multi-persona navigation instructions
"""

import os
import re
import cv2
import json
import math
import time
import base64
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ============================================================
# Third-party imports
# ============================================================
from openai import AzureOpenAI, OpenAI
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.tools.ImgTools import draw_landmarks, draw_star, crop_height, crop_rpg, crop_trajectory
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, Point2D, Point3D, view_area_corners
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose


# ============================================================
# Configuration Loading
# ============================================================
def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml
                     in the same directory as this script.

    Returns:
        Configuration dictionary
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please copy config.yaml.example to config.yaml and fill in your values."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# Configuration Class
# ============================================================
@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Data paths
    citynav_data_path: str = ""
    citynav_data_info_path: str = ""
    data_dir: str = "./LandmarkPhotoData"

    # Output paths
    landmark_save_path: str = "./output/landmark_data.json"
    landmark_revised_path: str = "./output/landmark_data_revised.json"
    instruction_save_path: str = "./output/instruction_persona.json"
    instruction_filted_path: str = "./output/instructions_filted.json"

    # Style configuration paths
    styles_path: str = "./config/styles.json"
    few_shot_path: str = "./config/few_shot.json"

    # Processing parameters
    action_num: int = 150
    max_landmark_distance: int = 1300
    min_landmark_distance: int = 200
    max_workers: int = 20

    # Start point sampling parameters
    min_start_target_distance: int = 300
    max_start_target_distance: int = 1500
    start_sampling_attempts: int = 100

    # Retry configuration
    max_retries: int = 5
    retry_delay: float = 1.0

    @classmethod
    def from_yaml(cls, config_path: str = None) -> "PipelineConfig":
        """Create config from YAML file"""
        config = load_config(config_path)
        paths = config.get("paths", {})
        params = config.get("params", {})

        return cls(
            citynav_data_path=paths.get("citynav_data_path", ""),
            citynav_data_info_path=paths.get("citynav_data_info_path", ""),
            data_dir=paths.get("data_dir", "./LandmarkPhotoData"),
            landmark_save_path=paths.get("landmark_save_path", "./output/landmark_data.json"),
            landmark_revised_path=paths.get("landmark_revised_path", "./output/landmark_data_revised.json"),
            instruction_save_path=paths.get("instruction_save_path", "./output/instruction_persona.json"),
            instruction_filted_path=paths.get("instruction_filted_path", "./output/instructions_filted.json"),
            styles_path=paths.get("styles_path", "./config/styles.json"),
            few_shot_path=paths.get("few_shot_path", "./config/few_shot.json"),
            action_num=params.get("action_num", 150),
            max_landmark_distance=params.get("max_landmark_distance", 1300),
            min_landmark_distance=params.get("min_landmark_distance", 200),
            max_workers=params.get("max_workers", 20),
            min_start_target_distance=params.get("min_start_target_distance", 300),
            max_start_target_distance=params.get("max_start_target_distance", 1500),
            start_sampling_attempts=params.get("start_sampling_attempts", 100),
            max_retries=params.get("max_retries", 5),
            retry_delay=params.get("retry_delay", 1.0),
        )


# ============================================================
# API Client Manager
# ============================================================
class APIClientManager:
    """API client manager for LLM services"""

    def __init__(self, config_path: str = None):
        """
        Initialize API clients from configuration file.

        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        self.gpt_clients: List[AzureOpenAI] = []

        config = load_config(config_path)
        api_config = config.get("api", {})
        self._init_clients(api_config)

    def _init_clients(self, api_config: dict):
        """Initialize all API clients from config"""
        # Initialize GPT-4o clients
        gpt4o_configs = api_config.get("gpt4o", [])
        for cfg in gpt4o_configs:
            client = AzureOpenAI(
                api_key=cfg["api_key"],
                api_version=cfg["api_version"],
                azure_endpoint=cfg["azure_endpoint"],
            )
            self.gpt_clients.append(client)

    def get_gpt_client(self, index: int) -> AzureOpenAI:
        """Get GPT-4o client (round-robin)"""
        return self.gpt_clients[index % len(self.gpt_clients)]


# ============================================================
# Utility Functions
# ============================================================
class Utils:
    """Common utility functions"""

    ACTION_DICT = {
        'MOVE_FORWARD': 1,
        'TURN_LEFT': 3,
        'TURN_RIGHT': 2,
        'STOP': 0
    }

    @staticmethod
    def safe_parse_json(s: str) -> list:
        """Safely parse JSON string"""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', s, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return []
        return []

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def to_actions_list(actions) -> List[int]:
        """Convert action objects to action ID list"""
        return [Utils.ACTION_DICT[action.name] for action in actions]

    @staticmethod
    def to_actions_names(actions) -> List[str]:
        """Convert action objects to action name list"""
        return [action.name for action in actions]

    @staticmethod
    def compute_pose(start_pose: Pose4D, predicted_px: List[int],
                     true_start_px: List[int], map_name: str) -> Pose4D:
        """Compute world pose from pixel coordinates"""
        if predicted_px == [0, 0]:
            return start_pose

        dx = predicted_px[0] - true_start_px[0]
        dy = predicted_px[1] - true_start_px[1]
        world_x = dx / 10 + start_pose.x
        world_y = start_pose.y - dy / 10
        base_pose = Pose4D(world_x, world_y, 66.05, 0)

        corners = view_area_corners(base_pose, GROUND_LEVEL[map_name])
        depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
        center_depth = depth_img[45:55, 45:55].mean()
        refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
        return refined_pose

    @staticmethod
    def move(pose: Pose4D, dst: Pose4D, iterations: int) -> Tuple[list, list]:
        """Compute trajectory and actions from current to target position"""
        dst_point = Point3D(dst.x, dst.y, pose.z)
        trajectory = []
        actions = []

        for _ in range(iterations):
            action = lookahead_discrete_action(pose, [dst_point])
            if action.name == 'STOP':
                return trajectory, actions
            pose = _moved_pose(pose, *action.value)
            trajectory.append(pose)
            actions.append(action)

        return trajectory, actions

    @staticmethod
    def calculate_distance(pos1: List[int], pos2: List[int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    @staticmethod
    def draw_surrounding(ori_image: np.ndarray, surrounding_coordinates: List,
                         save_path: str) -> None:
        """Draw surrounding boundary on image"""
        ori_image = Image.fromarray(ori_image)
        draw = ImageDraw.Draw(ori_image)

        polygon_points = [tuple(pt) for pt in surrounding_coordinates]
        draw.polygon(polygon_points, outline="red", fill=None)

        for i in range(len(polygon_points)):
            p1 = polygon_points[i]
            p2 = polygon_points[(i + 1) % len(polygon_points)]
            draw.line([p1, p2], fill=(255, 0, 0), width=8)

        ori_image.save(save_path)

    @staticmethod
    def px_to_pose(px: List[int], reference_pose: Pose4D, reference_px: List[int],
                   map_name: str) -> Pose4D:
        """Convert pixel coordinates to world pose"""
        dx = px[0] - reference_px[0]
        dy = px[1] - reference_px[1]
        world_x = dx / 10 + reference_pose.x
        world_y = reference_pose.y - dy / 10

        base_pose = Pose4D(world_x, world_y, 66.05, 0)

        try:
            depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
            center_depth = depth_img[45:55, 45:55].mean()
            refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
            return refined_pose
        except:
            return base_pose


# ============================================================
# Prompt Builder
# ============================================================
class PromptBuilder:
    """Prompt builder for constructing LLM prompts for each step"""

    @staticmethod
    def build_target_description_prompt(image_path: str, target_px: List[int],
                                          surrounding_info: str = "") -> List[dict]:
        """
        Step 1: Build prompt for target description generation
        Generate natural language description of target from endpoint view
        """
        image_b64 = Utils.encode_image_base64(image_path)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that generates clear and unambiguous descriptions "
                    "for navigation targets in drone aerial images. "
                    "Focus on unique visual features that can be reliably identified from aerial view. "
                    "The description should be concise (1-2 sentences) and avoid ambiguous terms."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": (
                        f"This is an aerial/satellite image. The red box marks the target location at pixel coordinates {target_px}.\n\n"
                        "Please generate a clear, concise description (1-2 sentences) of the target that:\n"
                        "1. Focuses on unique visual features visible from above (shape, color, size, texture)\n"
                        "2. Avoids ambiguous terms that could match multiple objects in the scene\n"
                        "3. Does NOT use relative directions (north, south, left, right)\n"
                        "4. Does NOT mention the red box or any annotations\n"
                        "5. Can be used to uniquely identify this target from a drone's perspective\n\n"
                        "Output ONLY the description, nothing else."
                    )}
                ]
            }
        ]
        return messages

    @staticmethod
    def build_target_ambiguity_filter_prompt(target_view_path: str,
                                              target_description: str) -> List[dict]:
        """
        Step 1 Quality Filter: Check if target description is ambiguous
        Filter samples with vague or confusing descriptions
        """
        image_b64 = Utils.encode_image_base64(target_view_path)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that evaluates whether a target description is clear and unambiguous. "
                    "You need to check if the description could potentially match multiple objects in the scene."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": (
                        f"Target description: \"{target_description}\"\n\n"
                        "Look at this aerial image and evaluate the target description:\n"
                        "1. Is the description clear and specific?\n"
                        "2. Could this description match TWO OR MORE distinct objects in the visible area?\n"
                        "3. Would a drone be able to uniquely identify the target using this description?\n\n"
                        "Rules:\n"
                        "- Output 'PASS' if the description is clear and unambiguous (only one object matches)\n"
                        "- Output 'FAIL' if the description is ambiguous (multiple objects could match)\n\n"
                        "Output ONLY 'PASS' or 'FAIL', nothing else."
                    )}
                ]
            }
        ]
        return messages

    @staticmethod
    def build_landmark_planning_prompt(image_path: str, start_px: List[int],
                                        target_px: List[int],
                                        target_description: str) -> List[dict]:
        """
        Step 2: Build prompt for landmark planning
        Identify intermediate landmarks along the path
        """
        image_b64 = Utils.encode_image_base64(image_path)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that extracts landmarks from drone satellite images. "
                    "Each landmark must include its exact coordinates and a brief description "
                    "that matches the visible feature at those coordinates. "
                    "Only output in the requested JSON format."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": (
                        f"You are given a satellite image. The drone's starting coordinates are {start_px} "
                        f"and the target coordinates are {target_px}. "
                        f"The red box highlights the {target_description}, which is the target area. "
                        "The green arrow indicates the drone's starting position and its current heading. "
                        "The yellow box represents the drone's field of view.\n\n"
                        "Your task is to select a set of landmarks that guide the drone step by step "
                        "from the start to the target.\n\n"
                        "Process for each landmark:\n"
                        "1. Choose a pixel coordinate [x, y] that lies close to the straight line path "
                        "between the start and the target.\n"
                        "2. Provide a short, precise description of its most visible and unique features.\n\n"
                        "Constraints:\n"
                        "1. Do not include the starting position.\n"
                        "2. The target position must be the final landmark.\n"
                        "3. Do not mention the red box, green arrow, or yellow box.\n"
                        "4. Use a small number of landmarks (2-6), ensuring each can appear within "
                        "the drone's moving field of view.\n"
                        "5. Descriptions must precisely match the object at the coordinates.\n"
                        "6. Do not use relative directions.\n\n"
                        "Output strictly in the following format:\n"
                        '[{"landmark_pos":[x1,y1], "landmark_description":"..."}, ...]'
                    )}
                ]
            }
        ]
        return messages

    @staticmethod
    def build_landmark_refinement_prompt(landmark_view_path: str,
                                          landmark_description: str) -> List[dict]:
        """
        Step 2 (cont.): Build prompt for landmark semantic refinement
        Verify and correct landmark description accuracy
        """
        image_b64 = Utils.encode_image_base64(landmark_view_path)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that checks whether a landmark description "
                    "matches a visible object in an image. If not, you will suggest a new description."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": (
                        f"Landmark description: {landmark_description}\n\n"
                        "Your task:\n"
                        "1. Read the given landmark description.\n"
                        "2. Look at the provided image (drone top-down view on a satellite map).\n"
                        "3. Decide if the described landmark can be identified in the CENTER of the image.\n\n"
                        "Rules:\n"
                        "- If the landmark is visible and identifiable in the center, output: YES\n"
                        "- If NOT visible or does not match, output:\n"
                        "  NO\n"
                        "  NEW_DESCRIPTION: <A concise description of the object at the center. "
                        "Do not mention 'center', 'image', or camera perspective.>\n\n"
                        "Follow the rules strictly."
                    )}
                ]
            }
        ]
        return messages

    @staticmethod
    def build_instruction_generation_prompt(data: dict, style: dict,
                                             few_shot: dict) -> List[dict]:
        """
        Step 4: Build prompt for instruction generation
        Generate natural and diverse navigation instructions
        """
        tra_map_b64 = Utils.encode_image_base64(data["tra_map_with_marked"])

        # Build landmark information
        landmarks_info = data["landmarks"]
        landmarks = []
        for i, item in enumerate(landmarks_info):
            landmarks.append({
                f"landmark{i + 1}_description": item["landmark_description"],
                f"landmark{i + 1}_pos": item["landmark_pos"]
            })

        # Build action-landmark alignment info
        action_segment = data["action_segment"]
        landmark_action_align = ""
        if len(action_segment) > 1:
            landmark_action_align = f"From start to landmark 1, the actions are: {action_segment[0]}.\n"
            for i in range(1, len(landmarks) - 1):
                landmark_action_align += f"From landmark {i} to landmark {i+1}, the actions are: {action_segment[i]}.\n"
            landmark_action_align += f"From landmark {len(landmarks)-1} to landmark {len(landmarks)} (target), the actions are: {action_segment[-1]}.\n"
        else:
            landmark_action_align = f"From start to landmark 1 (target), the actions are: {action_segment[0]}.\n"

        # Extract style parameters
        persona = style["role"]
        age_group = style["age"]
        background = style["background"]
        navigation_style = style["navigation_style"]

        # Few-shot examples
        example_1 = few_shot["easy"]["rewritten"]
        example_2 = few_shot["medium"]["rewritten"]
        example_3 = few_shot["hard"]["rewritten"]

        system_prompt = f"""
You are an assistant specializing in writing navigation instructions.

Your task is to generate one continuous, fluent navigation instruction that guides the drone from
the starting position, past each landmark in order, and ends precisely at the target.

=== Core Task Rules ===
1. Use only the available actions: "turn left", "turn right", "move forward", and "stop".
2. Do not mention coordinates or numbers.
3. The instruction must follow the provided trajectory.
4. Merge repeated or consecutive actions into smooth, natural phrases.
5. All turning instructions are relative to the drone's current heading.

=== Persona Style Control ===
You must write the navigation instruction in the linguistic style of the target persona.
Match the persona's tone, vocabulary level, pacing, level of detail, and sentence structure.

=== Output Requirement ===
Output only the navigation instruction written in the target persona's style.
"""

        user_prompt = f"""
=== Target Persona Description ===
Persona role: **{persona}**
Age group: {age_group}
Background: {background}
Navigation style: {navigation_style}

=== Few-shot Examples from This Persona ===
1. "{example_1}"
2. "{example_2}"
3. "{example_3}"

=== Navigation Context ===
Trajectory map: (see image - red arrow = start, green arrow = end, red dots = landmarks)

Landmark information:
{landmarks}

Starting pixel coordinates: {data["start_px"]}

Available actions: turn left (30°), turn right (30°), move forward, stop.

Action sequence between landmarks:
{landmark_action_align}

=== Your Task ===
Generate one continuous, natural navigation instruction that:
- Guides the drone past all landmarks in order
- Follows the trajectory map
- Merges repeated actions into smooth phrases
- Ends with "stop" at the target

=== Hard Constraints ===
1. Do not use numbers or coordinates.
2. Mention landmarks in the exact order given.
3. All turning instructions must be relative to the drone's heading.
4. End with "stop" at the target.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tra_map_b64}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        return messages


# ============================================================
# Step 1: Start & Target Selection
# ============================================================
class StartTargetSelector:
    """
    Step 1: Start & Target Selection Module

    Responsibilities:
    1. Randomly sample a valid coordinate as navigation start position
    2. Select target object with clear spatial boundaries
    3. Use MLLM to generate natural language description of target
    4. Filter samples with vague or confusing descriptions
    """

    def __init__(self, api_manager: APIClientManager, config: PipelineConfig):
        self.api_manager = api_manager
        self.config = config

    def random_sample_start(self, nav_gym: NavGym, target_px: List[int],
                            data_info: dict) -> Optional[Tuple[Pose4D, List[int]]]:
        """
        Randomly sample a valid start position on the map

        Args:
            nav_gym: Navigation environment instance
            target_px: Target pixel coordinates
            data_info: Data info (including map size, etc.)

        Returns:
            (start_pose, start_px) or None if sampling fails
        """
        # Get map dimensions
        image_size = data_info.get("image_size", [4000, 4000])
        map_width, map_height = image_size[0], image_size[1]

        # Get reference pose for coordinate conversion
        reference_pose = nav_gym.start_pose  # Original start as reference
        reference_px = nav_gym._get_px(reference_pose)
        map_name = nav_gym.episode.id[0]

        # Margin (avoid sampling at map edges)
        margin = 200

        for attempt in range(self.config.start_sampling_attempts):
            # Randomly sample pixel coordinates
            sampled_x = random.randint(margin, map_width - margin)
            sampled_y = random.randint(margin, map_height - margin)
            sampled_px = [sampled_x, sampled_y]

            # Check distance to target
            distance = Utils.calculate_distance(sampled_px, target_px)
            if distance < self.config.min_start_target_distance:
                continue  # Too close, resample
            if distance > self.config.max_start_target_distance:
                continue  # Too far, resample

            # Convert to world coordinates
            try:
                sampled_pose = Utils.px_to_pose(sampled_px, reference_pose, reference_px, map_name)

                # Verify position is valid (by getting depth map)
                depth_img = cropclient.crop_image(map_name, sampled_pose, (100, 100), "depth")
                center_depth = depth_img[45:55, 45:55].mean()

                # If depth is abnormal (NaN or too large), position is invalid
                if np.isnan(center_depth) or center_depth > 100:
                    continue

                # Randomly set heading (yaw)
                random_yaw = random.uniform(0, 2 * math.pi)
                sampled_pose = Pose4D(sampled_pose.x, sampled_pose.y, sampled_pose.z, random_yaw)

                print(f"  [Random Sampling] Start sampled: px={sampled_px}, distance_to_target={distance:.0f}")
                return sampled_pose, sampled_px

            except Exception as e:
                continue  # Position invalid, continue sampling

        print(f"  [Random Sampling] Sampling failed, using original start")
        return None

    def generate_target_description(self, target_view_path: str, target_px: List[int],
                                     index: int) -> Optional[str]:
        """
        Use MLLM to generate target description from endpoint view

        Args:
            target_view_path: Target view image path
            target_px: Target pixel coordinates
            index: Index for client round-robin

        Returns:
            Generated target description, or None if failed
        """
        client = self.api_manager.get_gpt_client(index)
        messages = PromptBuilder.build_target_description_prompt(
            image_path=target_view_path,
            target_px=target_px
        )

        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                description = response.choices[0].message.content.strip()

                if description and len(description) > 10:
                    print(f"  [MLLM] Target description: {description[:60]}...")
                    return description

            except Exception as e:
                print(f"  [MLLM] Target description failed (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay)

        return None

    def filter_ambiguous_target(self, target_view_path: str, target_description: str,
                                 index: int) -> bool:
        """
        Quality filter - filter samples with vague or confusing descriptions

        Args:
            target_view_path: Target view image path
            target_description: Target description
            index: Index for client round-robin

        Returns:
            True = passed (clear), False = filtered (ambiguous)
        """
        client = self.api_manager.get_gpt_client(index)
        messages = PromptBuilder.build_target_ambiguity_filter_prompt(
            target_view_path=target_view_path,
            target_description=target_description
        )

        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                result = response.choices[0].message.content.strip().upper()

                if "PASS" in result:
                    print(f"  [Quality Filter] Passed - clear description")
                    return True
                elif "FAIL" in result:
                    print(f"  [Quality Filter] Failed - ambiguous description")
                    return False

            except Exception as e:
                print(f"  [Quality Filter] Check failed (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay)

        # Default pass
        return True

    def select_start_and_target(self, nav_gym: NavGym, data_info: dict,
                                 index: int) -> Optional[dict]:
        """
        Execute full Step 1: Start & Target Selection

        Process:
        1. Get target position and boundary
        2. Random sample start point
        3. Save target view image
        4. Use MLLM to generate target description
        5. Quality filtering

        Args:
            nav_gym: Navigation environment instance
            data_info: Data info dictionary
            index: Processing index

        Returns:
            Dict with start, target, description info, or None if failed
        """
        episode_id = nav_gym.episode_id
        print(f"\n[Step 1] Start & Target Selection - Episode: {episode_id}")

        # 1. Get target position
        target_px = nav_gym.target_px
        print(f"  [Target] target_px={target_px}")

        # 2. Random sample start point (as per paper requirements)
        sampling_result = self.random_sample_start(nav_gym, target_px, data_info)

        if sampling_result is not None:
            start_pose, start_px = sampling_result
            # Note: nav_gym.start_pose is read-only
            # Save sampled start to return data for subsequent steps
        else:
            # Sampling failed, use original start
            start_pose = nav_gym.start_pose
            start_px = nav_gym._get_px(start_pose)
            print(f"  [Start] Using original start: start_px={start_px}")

        # 3. Get and save target view image
        # Move near target to get view
        map_name = nav_gym.episode.id[0]
        target_pose = Utils.px_to_pose(target_px, start_pose, start_px, map_name)

        # Save target view
        target_view_dir = os.path.join(nav_gym.father_image_dir, "target_view")
        os.makedirs(target_view_dir, exist_ok=True)
        target_view_path = os.path.join(target_view_dir, "target_view.jpg")

        try:
            target_rgb = cropclient.crop_image(map_name, target_pose, (448, 448), "rgb")
            cv2.imwrite(target_view_path, cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"  [Warning] Cannot get target view: {e}")
            # Use global map with target annotation
            _, whole_map = nav_gym._get_cur_trajectory_map()
            surrounding_path = nav_gym.father_image_dir + "/whole_map_surroundings.jpg"
            Utils.draw_surrounding(
                whole_map,
                data_info[episode_id]["surrounding_coordinates"],
                surrounding_path
            )
            target_view_path = surrounding_path

        # 4. Use MLLM to generate target description (as per paper requirements)
        print(f"  [MLLM] Generating target description...")
        generated_description = self.generate_target_description(
            target_view_path=target_view_path,
            target_px=target_px,
            index=index
        )

        if generated_description is None:
            # If MLLM fails, use original description as fallback
            generated_description = data_info[episode_id].get(
                "target_processed_description",
                "building"
            )
            print(f"  [Fallback] Using original description: {generated_description}")

        # 5. Quality filtering (as per paper requirements)
        print(f"  [Quality Filter] Checking description clarity...")
        is_clear = self.filter_ambiguous_target(
            target_view_path=target_view_path,
            target_description=generated_description,
            index=index
        )

        if not is_clear:
            print(f"  [Quality Filter] Sample filtered, skipping")
            return None

        # 6. Draw annotated global map
        _, whole_map = nav_gym._get_cur_trajectory_map()
        surrounding_path = nav_gym.father_image_dir + "/whole_map_surroundings.jpg"
        Utils.draw_surrounding(
            whole_map,
            data_info[episode_id]["surrounding_coordinates"],
            surrounding_path
        )

        # Build return data
        result = {
            "episode_id": episode_id,
            "start_pose": start_pose,
            "start_px": start_px,
            "target_px": target_px,
            "target_description": generated_description,  # MLLM generated description
            "target_view_path": target_view_path,
            "whole_map_surroundings": surrounding_path,
            "cur_whole_map": nav_gym.cur_whole_map,
            "cur_rgb_drone": nav_gym.cur_rgb_drone,
            "quality_filter_passed": True,
        }

        print(f"  [Step 1 Done] start={start_px}, target={target_px}")
        return result


# ============================================================
# Step 2: Landmark Planning
# ============================================================
class LandmarkPlanner:
    """
    Step 2: Landmark Planning Module

    Responsibilities:
    - Plan intermediate navigation landmarks between start and target
    - Apply distance constraints between adjacent landmarks
    - Refine landmark descriptions semantically for accuracy
    """

    def __init__(self, api_manager: APIClientManager, config: PipelineConfig):
        self.api_manager = api_manager
        self.config = config

    def plan_landmarks(self, base_data: dict, index: int) -> Optional[List[dict]]:
        """
        Plan intermediate landmarks from start to target

        Args:
            base_data: Base data generated from Step 1
            index: Index for client round-robin

        Returns:
            List of landmarks, each containing landmark_pos and landmark_description
        """
        print(f"\n[Step 2] Landmark Planning - Episode: {base_data['episode_id']}")

        client = self.api_manager.get_gpt_client(index)
        messages = PromptBuilder.build_landmark_planning_prompt(
            image_path=base_data["whole_map_surroundings"],
            start_px=base_data["start_px"],
            target_px=base_data["target_px"],
            target_description=base_data["target_description"]
        )

        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                content = response.choices[0].message.content

                # Validate format
                if not self._validate_landmark_format(content, base_data["target_px"]):
                    print(f"  [Landmark Planning] Invalid format, retrying...")
                    continue

                landmarks = Utils.safe_parse_json(content)

                # Validate distance constraints
                if not self._validate_distance_constraints(landmarks, base_data["start_px"]):
                    print(f"  [Landmark Planning] Distance constraint violated, retrying...")
                    continue

                # Filter landmarks that are too close
                landmarks = self._filter_close_landmarks(
                    landmarks, base_data["start_px"], base_data["target_px"]
                )

                print(f"  [Landmark Planning] Successfully planned {len(landmarks)} landmarks")
                return landmarks

            except Exception as e:
                print(f"  [Landmark Planning] Error (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay)

        return None

    def refine_landmark_descriptions(self, landmarks: List[dict], index: int) -> int:
        """
        Semantic refinement: verify and correct landmark descriptions

        Args:
            landmarks: List of landmarks
            index: Index for client round-robin

        Returns:
            Number of landmarks that were corrected
        """
        print(f"  [Semantic Refinement] Verifying landmark descriptions...")

        client = self.api_manager.get_gpt_client(index)
        revise_count = 0

        # Last landmark is the target, no need to modify
        for k in range(len(landmarks) - 1):
            landmark = landmarks[k]

            if "landmark_view" not in landmark:
                continue

            messages = PromptBuilder.build_landmark_refinement_prompt(
                landmark_view_path=landmark["landmark_view"],
                landmark_description=landmark["landmark_description"]
            )

            for attempt in range(self.config.max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                    )
                    content = response.choices[0].message.content

                    if content.upper().startswith("NO"):
                        # Extract new description
                        if "NEW_DESCRIPTION:" in content:
                            new_desc = content.split("NEW_DESCRIPTION:")[-1].strip()
                            landmark["landmark_description_old"] = landmark["landmark_description"]
                            landmark["landmark_description"] = new_desc
                            landmark["revised"] = True
                            revise_count += 1
                    else:
                        landmark["revised"] = False
                    break

                except Exception as e:
                    print(f"    Landmark {k+1} refinement failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.config.retry_delay)

        # Mark the last landmark
        if landmarks:
            landmarks[-1]["revised"] = False

        print(f"  [Semantic Refinement] Corrected {revise_count} landmark descriptions")
        return revise_count

    def _validate_landmark_format(self, response: str, target_px: List[int]) -> bool:
        """Validate if landmark format is correct"""
        landmarks = Utils.safe_parse_json(response)

        if not landmarks or not isinstance(landmarks, list):
            return False

        for item in landmarks:
            if not isinstance(item, dict):
                return False
            if "landmark_description" not in item or "landmark_pos" not in item:
                return False
            if not isinstance(item["landmark_description"], str):
                return False
            pos = item["landmark_pos"]
            if not isinstance(pos, list) or len(pos) != 2:
                return False
            if not all(isinstance(x, int) for x in pos):
                return False

        # Last landmark must be the target position
        if landmarks[-1]["landmark_pos"] != target_px:
            return False

        return True

    def _validate_distance_constraints(self, landmarks: List[dict], start_px: List[int]) -> bool:
        """Validate distance constraints between adjacent landmarks"""
        all_points = [start_px] + [lm["landmark_pos"] for lm in landmarks]

        for i in range(len(all_points) - 1):
            dist = Utils.calculate_distance(all_points[i], all_points[i + 1])
            if dist > self.config.max_landmark_distance:
                return False

        return True

    def _filter_close_landmarks(self, landmarks: List[dict], start_px: List[int],
                                 target_px: List[int]) -> List[dict]:
        """Filter landmarks that are too close to each other"""
        filtered = []
        cur_px = start_px

        # Process all landmarks except the last one
        for landmark in landmarks[:-1]:
            pos = landmark["landmark_pos"]
            dist = Utils.calculate_distance(pos, cur_px)
            if dist > self.config.min_landmark_distance:
                filtered.append(landmark)
                cur_px = pos

        # Check if the last filtered landmark is too close to target
        if filtered:
            last_pos = filtered[-1]["landmark_pos"]
            dist_to_target = Utils.calculate_distance(last_pos, target_px)
            if dist_to_target < self.config.min_landmark_distance:
                filtered.pop()

        # Always keep target as the last landmark
        filtered.append(landmarks[-1])

        return filtered


# ============================================================
# Step 3: Trajectory Synthesis
# ============================================================
class TrajectorySynthesizer:
    """
    Step 3: Trajectory Synthesis Module

    Responsibilities:
    - Generate action sequences for each pair of adjacent nodes (Look-Ahead strategy)
    - Concatenate all segment action sequences into complete trajectory
    - Save field-of-view images for each landmark
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def synthesize_trajectory(self, nav_gym: NavGym, base_data: dict,
                               landmarks: List[dict]) -> dict:
        """
        Synthesize complete trajectory from start to target

        Args:
            nav_gym: Navigation environment instance
            base_data: Base data (including randomly sampled start point)
            landmarks: List of landmarks

        Returns:
            Dictionary containing trajectory, action sequence, view images, etc.
        """
        print(f"\n[Step 3] Trajectory Synthesis - Episode: {base_data['episode_id']}")

        # Use randomly sampled start point (from Step 1)
        start_pose = base_data["start_pose"]
        start_px = base_data["start_px"]
        map_name = nav_gym.episode.id[0]

        # Set NavGym's current position to randomly sampled start
        # Note: nav_gym.cur_pose is read-only property, but nav_gym.cur_pos is writable internal attribute
        # This doesn't modify navgym source code, just sets instance attribute at runtime
        nav_gym.cur_pos = start_pose

        # Build list of all node pixel coordinates
        tar_px_list = [start_px]
        for landmark in landmarks:
            tar_px_list.append(landmark["landmark_pos"])

        # Initialize
        action_segment = []
        cur_trajectory = [start_pose]
        all_actions = []
        tar_pose_list = [start_pose]

        # Generate action sequence for each pair of adjacent nodes
        for k in range(len(tar_px_list) - 1):
            seg_start_px = tar_px_list[k]
            seg_target_px = tar_px_list[k + 1]

            # Compute target pose
            pred_pose = Utils.compute_pose(
                tar_pose_list[k], seg_target_px, seg_start_px, map_name
            )
            tar_pose_list.append(pred_pose)

            # Use Look-Ahead strategy to generate action sequence
            begin_pose = tar_pose_list[k]
            end_pose = tar_pose_list[k + 1]
            move_trajectory, move_actions = Utils.move(
                begin_pose, end_pose, self.config.action_num
            )

            # Execute actions
            nav_gym.step_times(Utils.to_actions_list(move_actions))

            # Update actual reached position
            tar_pose_list[k + 1] = nav_gym.cur_pose
            tar_px_list[k + 1] = nav_gym._get_px(nav_gym.cur_pose)

            # Record action sequence
            action_segment.append(Utils.to_actions_names(move_actions))
            cur_trajectory.extend(move_trajectory)
            all_actions.extend(move_actions)

            # Save landmark field-of-view image
            _, cur_view = nav_gym._get_cur_drone_view(keep_rgb=False)
            landmark_dir = os.path.join(nav_gym.father_image_dir, "landmark_view")
            os.makedirs(landmark_dir, exist_ok=True)
            landmark_view_path = os.path.join(landmark_dir, f"landmark_{k + 1}.jpg")
            cv2.imwrite(landmark_view_path, cur_view)
            landmarks[k]["landmark_view"] = landmark_view_path

        # Add STOP action
        action_segment[-1].append("STOP")

        # Generate trajectory map
        _, map_with_tra = nav_gym._get_cur_trajectory_map()
        tra_map_path = nav_gym.father_image_dir + "/map_with_tra.jpg"
        plt.imsave(tra_map_path, map_with_tra)

        # Draw trajectory map with markers
        marked_map_path = nav_gym.father_image_dir + "/tra_map_with_marked.jpg"
        self._draw_marked_trajectory_map(
            nav_gym, tra_map_path, marked_map_path, start_pose, landmarks
        )

        # Build result
        action_names = Utils.to_actions_names(all_actions)
        action_names.append("STOP")

        result = {
            **base_data,
            "landmarks": landmarks,
            "action_segment": action_segment,
            "map_with_tra": tra_map_path,
            "tra_map_with_marked": marked_map_path,
            "actions": action_names,
        }

        print(f"  [Trajectory Synthesis] Done: {len(landmarks)} landmarks, {len(action_names)} actions")
        return result

    def _draw_marked_trajectory_map(self, nav_gym: NavGym, tra_map_path: str,
                                     save_path: str, start_pose: Pose4D,
                                     landmarks: List[dict]) -> None:
        """Draw trajectory map with start point and landmark markers"""
        tra_image = cv2.imread(tra_map_path, cv2.IMREAD_UNCHANGED)

        # Draw start point arrow
        directions = [
            math.sin(math.pi / 2 + start_pose.yaw),
            math.cos(math.pi / 2 + start_pose.yaw)
        ]
        tra_image = crop_trajectory(
            image=tra_image,
            px_trajectory=[nav_gym._get_px(start_pose)],
            area=None,
            savefig=True,
            directions=directions,
            color=(0, 0, 255, 60)
        )

        # Draw landmark points
        for landmark in landmarks:
            cv2.circle(
                tra_image,
                tuple(landmark["landmark_pos"]),
                radius=5,
                color=(0, 0, 255),
                thickness=20
            )

        cv2.imwrite(save_path, tra_image)


# ============================================================
# Step 4: Instruction Generation
# ============================================================
class InstructionGenerator:
    """
    Step 4: Instruction Generation Module

    Responsibilities:
    - Generate natural language navigation instructions based on trajectory, map, and node info
    - Support multiple persona styles
    - Use few-shot examples for language style control
    """

    def __init__(self, api_manager: APIClientManager, config: PipelineConfig):
        self.api_manager = api_manager
        self.config = config
        self.styles = None
        self.few_shot = None

    def load_style_configs(self):
        """Load style configuration files"""
        with open(self.config.styles_path, 'r', encoding='utf-8') as f:
            self.styles = json.load(f)
        with open(self.config.few_shot_path, 'r', encoding='utf-8') as f:
            self.few_shot = json.load(f)

    def generate_instructions(self, trajectory_data: dict, index: int,
                               num_styles: int = 4) -> Dict[str, dict]:
        """
        Generate navigation instructions with multiple personas for a trajectory
        """
        print(f"\n[Step 4] Instruction Generation - Episode: {trajectory_data['episode_id']}")

        if self.styles is None:
            self.load_style_configs()

        client = self.api_manager.get_gpt_client(index)
        selected_styles = random.sample(self.styles, min(num_styles, len(self.styles)))

        results = {}
        for style_idx, style in enumerate(selected_styles):
            persona = style["role"]
            messages = PromptBuilder.build_instruction_generation_prompt(
                trajectory_data, style, self.few_shot[persona]
            )

            instruction = self._call_llm_with_retry(client, messages)
            if instruction:
                key = f"{trajectory_data['episode_id']}_{style_idx}"
                results[key] = {
                    "episode_id": trajectory_data["episode_id"],
                    "persona": persona,
                    "instruction": instruction,
                    "landmarks": trajectory_data["landmarks"],
                    "total_actions": trajectory_data["actions"],
                }
                print(f"  [Instruction Generation] Persona '{persona}': {instruction[:50]}...")

        return results

    def _call_llm_with_retry(self, client, messages: List[dict]) -> Optional[str]:
        """Call LLM with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"    Instruction generation error (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay)
        return None


# ============================================================
# Main Pipeline
# ============================================================
class DataPipeline:
    """
    AirNav Data Generation Main Pipeline

    Integrates all four steps and provides batch processing capability
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.api_manager = APIClientManager()

        # Initialize modules
        self.start_target_selector = StartTargetSelector(self.api_manager, self.config)
        self.landmark_planner = LandmarkPlanner(self.api_manager, self.config)
        self.trajectory_synthesizer = TrajectorySynthesizer(self.config)
        self.instruction_generator = InstructionGenerator(self.api_manager, self.config)

        # Load image cache
        cropclient.load_image_cache()

    def process_single_episode(self, index: int, citynav_data, data_info: dict,
                                processed_ids: set = None) -> Optional[dict]:
        """
        Process complete pipeline for a single episode

        Process:
        Step 1: Start & Target Selection
            - Random start point sampling
            - MLLM target description generation
            - Quality filtering
        Step 2: Landmark Planning
            - Landmark identification
            - Distance constraints
            - Semantic refinement
        Step 3: Trajectory Synthesis
            - Look-Ahead strategy
            - Trajectory concatenation
        """
        # Initialize navigation environment
        nav_gym = NavGym(citynav_data, data_dir=os.path.abspath(self.config.data_dir))
        episode_id = nav_gym.episode_id

        # Check if already processed
        if processed_ids and episode_id in processed_ids:
            print(f"Episode {episode_id} already processed, skipping")
            return None

        print(f"\n{'='*60}")
        print(f"Processing Episode: {episode_id}")
        print(f"{'='*60}")

        try:
            # ============================================================
            # Step 1: Start & Target Selection
            # ============================================================
            base_data = self.start_target_selector.select_start_and_target(
                nav_gym, data_info, index
            )

            if base_data is None:
                print(f"[Failed] Step 1 did not pass quality filter")
                return None

            # ============================================================
            # Step 2: Landmark Planning
            # ============================================================
            landmarks = self.landmark_planner.plan_landmarks(base_data, index)
            if landmarks is None:
                print(f"[Failed] Step 2 landmark planning failed")
                return None

            # ============================================================
            # Step 3: Trajectory Synthesis
            # ============================================================
            trajectory_data = self.trajectory_synthesizer.synthesize_trajectory(
                nav_gym, base_data, landmarks
            )

            # Step 2 continued: Semantic refinement (after obtaining landmark view images)
            self.landmark_planner.refine_landmark_descriptions(
                trajectory_data["landmarks"], index
            )

            print(f"\n[Done] Episode {episode_id} processed successfully")
            return trajectory_data

        except Exception as e:
            print(f"[Error] Episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_full_pipeline(self):
        """
        Run the complete four-step pipeline
        """
        print("=" * 60)
        print("AirNav Data Pipeline - Full Run")
        print("=" * 60)

        # Step 1-3: Landmark generation and trajectory synthesis
        print("\n[Step 1-3] Generating landmarks and trajectories...")
        self.run_landmark_generation()

        # Step 2 (continued): Landmark description refinement
        print("\n[Step 2+] Refining landmark descriptions...")
        self.run_landmark_refinement()

        # Step 4: Instruction generation
        print("\n[Step 4] Generating navigation instructions...")
        self.run_instruction_generation()

        print("\n" + "=" * 60)
        print("Pipeline completed!")
        print("=" * 60)

    def run_landmark_generation(self, save_path: str = None, max_workers: int = None,
                                 resume_from: str = None):
        """Run batch landmark generation (Step 1-3)"""
        save_path = save_path or self.config.landmark_save_path
        max_workers = max_workers or self.config.max_workers

        # Load data
        citynav_data = CityNavData(self.config.citynav_data_path)
        with open(self.config.citynav_data_info_path, 'r', encoding='utf-8') as f:
            data_info_list = json.load(f)

        # Convert to dictionary format
        data_info = {}
        for item in data_info_list:
            label = "_".join(str(x) for x in item['episode_id'])
            data_info[label] = item

        # Resume from checkpoint
        processed_ids = set()
        total_data = []
        if resume_from and os.path.exists(resume_from):
            with open(resume_from, 'r', encoding='utf-8') as f:
                total_data = json.load(f)
            processed_ids = {item['episode_id'] for item in total_data}
            print(f"Resuming from {len(total_data)} processed episodes")

        total = len(citynav_data)
        chunk_size = 1000

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = [citynav_data[k] for k in range(chunk_start, chunk_end)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.process_single_episode,
                        i, chunk[i], data_info, processed_ids
                    )
                    for i in range(len(chunk))
                ]

                with tqdm(total=len(chunk), desc=f"Landmarks [{chunk_start}-{chunk_end}]") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            total_data.append(result)
                            processed_ids.add(result['episode_id'])
                        pbar.update(1)

            # Save after each chunk
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(total_data, f, indent=4, ensure_ascii=False)
            print(f"Saved {len(total_data)} episodes to {save_path}")

    def run_landmark_refinement(self, landmark_path: str = None,
                                 save_path: str = None, max_workers: int = None):
        """Run batch landmark description refinement"""
        landmark_path = landmark_path or self.config.landmark_save_path
        save_path = save_path or self.config.landmark_revised_path
        max_workers = max_workers or 10

        with open(landmark_path, 'r', encoding='utf-8') as f:
            landmark_data = json.load(f)

        total = len(landmark_data)
        total_revise_count = 0

        def refine_single(index: int) -> int:
            return self.landmark_planner.refine_landmark_descriptions(
                landmark_data[index]["landmarks"], index
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(refine_single, i) for i in range(total)]

            with tqdm(total=total, desc="Refining landmarks") as pbar:
                for i, future in enumerate(as_completed(futures)):
                    total_revise_count += future.result()
                    pbar.update(1)

                    if (i + 1) % 1000 == 0:
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(landmark_data, f, indent=4, ensure_ascii=False)

        print(f"Total corrections: {total_revise_count}")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(landmark_data, f, indent=4, ensure_ascii=False)

    def run_instruction_generation(self, landmark_path: str = None,
                                    save_path: str = None, max_workers: int = None):
        """Run batch instruction generation (Step 4)"""
        landmark_path = landmark_path or self.config.landmark_revised_path
        save_path = save_path or self.config.instruction_save_path
        max_workers = max_workers or self.config.max_workers

        with open(landmark_path, 'r', encoding='utf-8') as f:
            landmark_data = json.load(f)

        all_instructions = {}
        total = len(landmark_data)

        def generate_single(index: int) -> Dict[str, dict]:
            return self.instruction_generator.generate_instructions(
                landmark_data[index], index
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_single, i) for i in range(total)]

            with tqdm(total=total, desc="Generating instructions") as pbar:
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result:
                        all_instructions.update(result)
                    pbar.update(1)

                    if (i + 1) % 1000 == 0:
                        with open(save_path, 'w', encoding='utf-8') as f:
                            json.dump(all_instructions, f, indent=4, ensure_ascii=False)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_instructions, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(all_instructions)} instructions to {save_path}")


# ============================================================
# Entry Point
# ============================================================
def main():
    """Main function"""
    config = PipelineConfig(
        max_workers=20,
    )

    pipeline = DataPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
