import re
import json
from pathlib import Path
import rasterio
from copy import deepcopy

def get_format_reward(response, ground_truth, extra_info):
    response = response.replace("'", '"')
    match =re.search(r'\[(.*?)\]', response, flags=re.S)
    if match:
        response = match.group(0)
    else:
        return 0.0
    allowed_actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    
    try:
        actions = json.loads(response)
        if (
            isinstance(actions, list)
            and actions
            and all(isinstance(a, str) and a in allowed_actions for a in actions)
            and len(actions) <= 8
        ):  
            for i in range(len(actions)):
                if actions[i] == "STOP" and i != len(actions) - 1:
                    return 0.0
                
            return 0.1
        else:
            return 0.0
    except json.JSONDecodeError:
        return 0.0
    #return 0.0

def get_action_reward(solution_str, ground_truth, extra_info):
    gt_actions = json.loads(ground_truth)
    solution_str = solution_str.replace("'", '"')
    match =re.search(r'\[(.*?)\]', solution_str, flags=re.S)
    weights = {
        "MOVE_FORWARD": 1,
        "TURN_LEFT": 3,
        "TURN_RIGHT": 3,
        "STOP": 7
    }
    if match:
        solution_str = match.group(0)
    actions = json.loads(solution_str)

    total_weight = 0
    sum_weight = 0
    
    stop_reward = 0.1
    gt_stop = False
    pre_stop = False
    
    for i in range(len(gt_actions)):
        if gt_actions[i] == "STOP":
            gt_stop = True
    
    for i in range(len(actions)):
        if actions[i] == "STOP":
            pre_stop = True
    
    if pre_stop == gt_stop:
        if gt_stop == True:
            stop_reward = 1
    else:
        stop_reward = 0
            
    return stop_reward


image_dir = Path("AirNav/data/rgbd-new")
raster_cache = {
    raster_path.stem: rasterio.open(raster_path)
    for raster_path in image_dir.glob("*.tif")
}
import math

def dump_yaw(yaw):
    return (yaw + math.pi) % (2 * math.pi) - math.pi

def steps(actions, cur_pos, is_degree=True):
    pos = cur_pos.copy()
    x = pos[0]
    y = pos[1]
    z = pos[2]
    yaw = pos[3]
    if is_degree:
        yaw = math.radians(yaw)
        
    for action in actions:
        if action == "MOVE_FORWARD":
            x = x + 5 * math.cos(yaw)
            y = y + 5 * math.sin(yaw)
        elif action == "TURN_LEFT": 
            yaw = dump_yaw(yaw + math.pi / 6)
        elif action == "TURN_RIGHT": 
            yaw = dump_yaw(yaw - math.pi / 6)
        elif action == "STOP": 
            break
        
    if is_degree:
        yaw = math.degrees(yaw)
        
    return [x, y, z, yaw]

def get_distance_reward(solution_str, ground_truth, extra_info):
    raster = raster_cache[extra_info["map_name"]]

    gt_actions = json.loads(ground_truth)
    solution_str = solution_str.replace("'", '"')
    match =re.search(r'\[(.*?)\]', solution_str, flags=re.S)
    if match:
        solution_str = match.group(0)
    actions = json.loads(solution_str)

    cur_pos = extra_info["cur_pose"]
    cur_px = extra_info["cur_px"]
    target_px = extra_info["target_px"]

    gt_pos = steps(gt_actions, cur_pos)
    pred_pos = steps(actions, cur_pos)

    raster = raster_cache[extra_info["map_name"]]

    gt_y, gt_x = raster.index(gt_pos[0], gt_pos[1])
    gt_x = int(gt_x)
    gt_y = int(gt_y)
    
    pred_y, pred_x = raster.index(pred_pos[0], pred_pos[1])
    pred_x = int(pred_x)
    pred_y = int(pred_y)

    ori_dis = math.sqrt((gt_x - cur_px[0]) ** 2 + (gt_y - cur_px[1]) ** 2)
    now_dis = math.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    
    eps = 1e-6
    ori_dis += eps
    dis_weight = (ori_dis - now_dis)/ori_dis
    
    yaw_diff = (gt_pos[3] - pred_pos[3] + 180) % 360 - 180
    
    yaw_weight = max(1 - abs(yaw_diff)/60 , 0)
    
    score = max(dis_weight * 1, 0) + max(yaw_weight * 1, 0)

    return score

def compute_score(data_source, solution_str, ground_truth, extra_info):
    action_reward = 0.0
    distance_reward = 0.0
    format_reward = get_format_reward(solution_str, ground_truth, extra_info)
    if format_reward > 0:
        action_reward = get_action_reward(solution_str, ground_truth, extra_info)
        distance_reward = get_distance_reward(solution_str, ground_truth, extra_info)
    reward = format_reward + action_reward + distance_reward
    
    return reward