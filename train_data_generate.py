import os
import sys
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
from navgym.models.AirNavData import AirNavData
from navgym.models.NavGym import NavGym
from navgym.tools.EvalTools import eval_planning_metrics
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, view_area_corners
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import json
from copy import deepcopy
import math



SAVE_PATH = "./experiment"
cropclient.load_image_cache()

action_dict = {
    'MOVE_FORWARD' : 1, 
    'TURN_LEFT' : 3,
    'TURN_RIGHT' : 2,
    'STOP': 0
}

def compute_pose(start_pose, predicted_px, true_start_px, map_name):
    if predicted_px == [0, 0]:
        return start_pose

    dx, dy = predicted_px[0] - true_start_px[0], predicted_px[1] - true_start_px[1]
    world_x = dx / 10 + start_pose.x
    world_y = start_pose.y - dy / 10
    base_pose = Pose4D(world_x, world_y, 66.05, 0)

    corners = view_area_corners(base_pose, GROUND_LEVEL[map_name])
    depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
    center_depth = depth_img[45:55, 45:55].mean()
    refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
    return refined_pose

from gsamllavanav.space import Point2D, Point3D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose


def move(pose: Pose4D, dst: Pose4D, iterations: int):

    dst = Point3D(dst.x, dst.y, pose.z)
    trajectory = []
    actions = []
    for _ in range(iterations):
        action = lookahead_discrete_action(pose, [dst])
        if action.name == 'STOP':
            return trajectory,actions
        pose = _moved_pose(pose, *action.value)
        trajectory.append(pose)
        actions.append(action)
    return trajectory,actions

def to_actions_list(actions):
    actions_list = []
    for action in actions:
        actions_list.append(action_dict[action])
    return actions_list

def to_actions_names(actions):
    actions_names = []
    for action in actions:
        actions_names.append(action.name)
    return actions_names

def position_to_pose5d(position):
    x, y, z, yaw_degree = position
    yaw = math.radians(yaw_degree)
    return [x, y, z, yaw, 0.0]

def generate_train_data_episode(key, airnavData, instruction):
    
    cur_trajectory = []
    
    cur_actions = []
    action_list = []

    generate_data_list = []
    
    cur_episode_id = instruction["episode_id"]
    
    cur_airnavData = airnavData

    navGym = NavGym(cur_airnavData,data_dir = os.path.abspath('./TrainPhotoData'))
    landmarks = instruction["landmarks"]
    persona = instruction["persona"]

    start_pose = navGym.start_pose
    map_name = navGym.episode.id[0]

    final_target_px = navGym.target_px
    true_start_px = navGym.px_trajectory[0]

    tar_px_list = [true_start_px]
    for k in range(len(landmarks)):
        tar_px_list.append(landmarks[k]["landmark_pos"])

    cur_trajectory = [start_pose]

    total_actions = instruction["total_actions"]

    action_num_per_step = 8

    count = math.ceil(len(total_actions) / action_num_per_step)
    
    
    tar_pose5d_list = [position_to_pose5d(navGym.cur_position)]
    history_view = []
    history_actions = []
    _ , start_view = navGym._get_cur_drone_view()

    for k in range(count):
        data_dict = dict()
        data_dict["total_actions"] = total_actions
        data_dict["persona"] = persona
        data_dict["episode_id_case"] = navGym.episode_id + f"_case{k}"
        data_dict["instruction"] = instruction["instruction"]
        data_dict["cur_position"] = navGym.cur_position
        data_dict["cur_position_px"] = navGym.cur_position_px
        data_dict["final_target_px"] = final_target_px
        data_dict["episode_id"] = navGym.episode_id
        data_dict["map_name"] = map_name
        data_dict["key"] = key

        save_path = navGym.father_image_dir + f"/case{k}"
        os.makedirs(save_path, exist_ok=True)

        data_dict["history_views"] = history_view.copy()

        _ , cur_view = navGym._get_cur_drone_view()

        plt.imsave(save_path + "/cur_view.jpg",cur_view)
        data_dict["cur_view"] = save_path + "/cur_view.jpg"

        history_view.append(save_path + "/cur_view.jpg")

        data_dict["history_actions"] = deepcopy(history_actions)

        l = k * action_num_per_step
        r = min((k + 1) * action_num_per_step, len(total_actions))
        
        data_dict["future_actions"] = total_actions[l:r]
        generate_data_list.append(data_dict)

        move_actions = total_actions[l:r]
        
        cur_actions.extend(move_actions)
        action_list.extend(to_actions_list(move_actions))
        move_actions_list = to_actions_list(move_actions)
        for act in move_actions_list:
            navGym.step(act)
            tar_pose5d_list.append(position_to_pose5d(navGym.cur_position))
            
        history_actions.extend(move_actions)

    return generate_data_list


GLOBAL_AIRNAV_DATA = None

def worker(key, idx, instruction):
    global GLOBAL_AIRNAV_DATA
    airnav_data = GLOBAL_AIRNAV_DATA[idx]
    return generate_train_data_episode(key, airnav_data, instruction)


def train_data_generate(chunk_size=2000):
    global GLOBAL_AIRNAV_DATA

    data_path = "./data/AirNav/train/airnav_train.json"
    instruction_path = "./data/AirNav/train/info_train.json"
    output_path = "./data/AirNav/train/train.json"

    with open(instruction_path, 'r') as f:
        instructions = json.load(f)

    keys = list(instructions.keys())
    total = len(keys)

    GLOBAL_AIRNAV_DATA = AirNavData(data_path)

    airnav_index = {}
    for k, item in enumerate(GLOBAL_AIRNAV_DATA):
        episode_id = item.episode.id[-1]
        airnav_index[episode_id] = k

    print("Index built.")

    open(output_path, "w").close()
    
    all_results = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_keys = keys[start:end]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for key in chunk_keys:
                inst = instructions[key]
                episode_id = inst["episode_id"]
                idx = airnav_index[episode_id]

                futures.append(executor.submit(worker, key, idx, inst))

            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"processing {start} ~ {end}"):
                data_list = f.result()
                if data_list:
                    all_results.extend(data_list)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
            
    print("All chunks done! Output:", output_path)

if __name__ == "__main__":
    train_data_generate()
