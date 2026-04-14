from openai import AzureOpenAI
import json
import base64
from navgym.models.AirNavData import AirNavData
import os
from navgym.models.NavGym import NavGym
import matplotlib.pyplot as plt
from navgym.tools.EvalTools import eval_planning_metrics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import re
import random


GPT_client = AzureOpenAI(
    api_key="EMPTY",
    api_version="2024-12-01-preview",
    azure_endpoint="your_azure_endpoint"
)
GPT_model = "gpt-4o"


Qwen_client = OpenAI(
    base_url="http://localhost:8000/v1",  
    api_key="EMPTY"                     
)
Qwen_model = "qwen_2_5_vl_7b"

MODEL_TYPE = Qwen_model
TEMPERATURE = 1.0

max_workers = 5  
save_file_name = "qwen_2_5_vl_7b_airnav_eval"

result_save_path1 = f"./result/val/{save_file_name}.json"
actions_save_path1 = f"./result/val/actions/{save_file_name}.json"
result_save_path2 = f"./result/test/{save_file_name}.json"
actions_save_path2 = f"./result/test/actions/{save_file_name}.json"

action_dict = {
    "MOVE_FORWARD" : 1, 
    "TURN_LEFT" : 3,
    "TURN_RIGHT" : 2,
    "STOP": 0
} # 3 -> + 2-> -

def to_actions_list(actions):
    actions_list = []
    for action in actions:
        actions_list.append(action_dict[action])
    return actions_list

def generate_random_action():
    weights = [38, 3, 3, 1]
    actions_list = list(action_dict.keys())
    actions = []
    
    while True:
        action = random.choices(actions_list, weights)[0]
        actions.append(action)
        if action == "STOP":
            break
    return actions

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_eval_messages(data):

    history_views_path = data["history_views"]
    cur_view_path = data["cur_view"]
    content_list = []
    cur_pose = str(data["cur_position"])
    instruction = data["instruction"]
    history_actions = data["history_actions"]
    
    user_prompt = f"""
## Text Input
- **Navigation instruction**: {instruction}  
- **Current state of the UAV**: {cur_pose} (x, y, z in meters; heading in degrees)
- **Previously executed actions**: {history_actions if history_actions else "None"}  
  (A list of past actions the UAV has taken, in chronological order.)

## Image Input
**UAV (Unmanned Aerial Vehicle) Top-Down View Sequence**  
- Historical top-down views (from oldest to newest) show the UAV’s past observations. 
- The last image is the current top-down view of the UAV.  
- In all images, the **top of the image corresponds to the UAV’s forward direction** (its heading).

Based on the navigation instruction, the UAV’s current state, the previously executed actions (which can help infer the UAV’s current orientation and progress), and the provided images, predict how the UAV should move **step by step** to follow the instruction accurately.

## Prediction Rules
1. Predict no more than **8 future actions** for the UAV to execute.
2. If the target location is reachable in fewer than 8 actions, output less than 8 actions sequence and end with **"STOP"**. Otherwise, it clearly requires more than 8 actions to approach the target, output exactly 8 future actions.
3. You **must** output **"STOP"** if the UAV has already reached the described target.
4. Output a **JSON list** of actions, in the **exact order** they should be executed.
5. Do **not** include any explanations, reasoning, or additional text — only output the JSON list.

## Discrete Action Space
- `MOVE_FORWARD`: move straight 5 meters in the current heading
- `TURN_LEFT`: rotate left 30°
- `TURN_RIGHT`: rotate right 30°
- `STOP`: stop the flight

## Output Format Examples
["TURN_RIGHT", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"]  
or  
["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]  
or  
["STOP"]
"""

    text_content = {"type": "text", "text": user_prompt}

    for i, path in enumerate(history_views_path, 1):
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
        })
        
        content_list.append({
            "type": "text",
            "text": f"(Above is historical view {i})"
        })
    
    content_list.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image(cur_view_path)}"},
    })
    content_list.append({"type": "text", "text": "(Above is the current view)"})

    content_list.append(text_content)

    system_prompt = """
## Role
You are an expert navigation assistant for a UAV (Unmanned Aerial Vehicle) flight simulator.

## Task Objective
The UAV operates in a 3D urban environment with visible roads, buildings, and landmarks.
Your task is to predict the next sequence of UAV actions based on:
1. A given natural language navigation instruction,
2. The current state of the UAV, including its position and heading angle,
3. The current top-down UAV view image,
4. Up to four historical top-down view images from previous time steps (if available),
5. The previously executed UAV actions (if available).
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    return messages


def parse_response(response):
    response = response.replace("'", '"')
    match =re.search(r'\[(.*?)\]', response, flags=re.S)
    if match:
        response = match.group(0)
    allowed_actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    try:
        actions = json.loads(response)
        if (
            isinstance(actions, list)
            and actions
            and all(isinstance(a, str) and a in allowed_actions for a in actions)
        ):
            return actions,True
        else:
            print(response)
            print("Response is not a valid list of strings.")
            return ["STOP"],False
    except json.JSONDecodeError:
        print(response)
        print("Failed to parse response as JSON.")
        return ["STOP"],False

def eval_one_episode(airnav_data, airnav_index, eval_data, key):

    cur_episode_id = eval_data[key]["episode_id"]
    if cur_episode_id not in airnav_index:
        print(f"Warning: {cur_episode_id} not in airnav_data")
        return None,None, None 
    
    navGym = NavGym(airnav_data[airnav_index[cur_episode_id]], data_dir=os.path.abspath('./EvalPhotoData'))
    history_views = []
    history_actions = []
    cur_pose = navGym.start_pose
    cur_trajectory = [cur_pose]
    total_steps = 0
    total_actions = []
    k = 0
    time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    

    while True:
        data_dict = dict()

        data_dict["episode_id_case"] = key + f"_case{k}"
        
        data_dict["instruction"] = eval_data[key]["instruction"]

        save_path = navGym.father_image_dir + f"/{time}_case_{k}"
        os.makedirs(save_path, exist_ok=True)

        data_dict["history_actions"] = history_actions.copy()

        data_dict["history_views"] = []
        
        indices = [-7, -4, -2, -1]
        for index in indices:
            if len(history_views) >= abs(index):
                data_dict["history_views"].append(history_views[index])

        _, cur_view = navGym._get_cur_drone_view()
        plt.imsave(save_path + "/cur_view.jpg", cur_view)
        data_dict["cur_view"] = save_path + "/cur_view.jpg"
        history_views.append(save_path + "/cur_view.jpg")

        data_dict["cur_position"] = navGym.cur_position
        message = get_eval_messages(data_dict)
        
        if MODEL_TYPE == Qwen_model:
            client = Qwen_client
        else:
            client = GPT_client
              
        flag = True
        retry_count = 0
        while not flag:
            try:
                retry_count += 1
                if retry_count > 2:
                    print(f"Max retries reached for episode {key}, defaulting to STOP.")
                    actions = ["STOP"]
                    break
                response = client.chat.completions.create(
                    model=MODEL_TYPE,
                    messages=message,
                    temperature=TEMPERATURE,
                )
                actions,flag = parse_response(response.choices[0].message.content)
                if not flag:
                    print(f"Error parsing response for episode {key}, retrying.")
            except Exception as e:
                print(f"Error during API call for episode {key}: {e}, retrying.")
                
        actions = generate_random_action()
        history_actions.extend(actions)
        actions_list = to_actions_list(actions)

        for action in actions_list:
            if action == 0:
                break
            cur_trajectory.append(navGym.cur_pose)
            navGym.step(action, savefig=False, saveviewfig=False)

        total_steps += len(actions_list)
        total_actions.extend(actions)

        cur_px = navGym.cur_position_px

        if actions[-1] == "STOP" or total_steps >= 160:  
            break
        k += 1
        
    _, map_with_tar = navGym._get_cur_trajectory_map()
    plt.imsave(navGym.father_image_dir + f"/{key}_final_map.jpg", map_with_tar)
    
    result_data = {
        "actions": total_actions,
        "end_position": navGym.cur_position,
        "end_position_px": navGym.cur_position_px,
        "total_steps": total_steps
    }
    return (key, airnav_data[airnav_index[cur_episode_id]].episode, cur_trajectory, result_data)

def eval_test(airnav_data, airnav_index, eval_data, max_workers):
    trajectories = dict()
    episodes = dict()
    action_records = dict()
    total = len(eval_data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(eval_one_episode, airnav_data, airnav_index, eval_data, key)
            for key in eval_data.keys()
        ]
        with tqdm(total=total) as pbar:
            for future in as_completed(futures):
                key, eid, traj, actions = future.result()
                if eid is not None:
                    episodes[key] = eid
                    trajectories[key] = traj
                    action_records[key] = actions
                pbar.update(1)

    return episodes, trajectories, action_records


def val():
    eval_data_paths = ["./data/AirNav/val/info_val_seen.json",
                       "./data/AirNav/val/info_val_unseen.json"]
    airnav_data_paths = ["./data/AirNav/val/airnav_val_seen.json",
                          "./data/AirNav/val/airnav_val_unseen.json"]
    eval_type = ["val_seen","val_unseen"]
    total_result = []
    
    total_trajectories = dict()
    total_episodes = dict()
    total_actions = dict()
    for i in range(2):
        
        print(f"================= Start eval {eval_type[i]} =================")
        airnav_data_path = airnav_data_paths[i]
        
        airnav_data = AirNavData(airnav_data_path)
        airnav_data_dict = dict()
        
        airnav_index = dict()
        for k, item in enumerate(airnav_data):
            episode_id = item.episode.id[-1]
            airnav_index[episode_id] = k
        
        eval_data_path = eval_data_paths[i]
        
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)

        episodes, trajectories, actions = eval_test(airnav_data, airnav_index, eval_data, max_workers=max_workers)
        
        total_actions[eval_type[i]] = actions
        total_trajectories.update(trajectories)
        total_episodes.update(episodes)
        
        metrics = eval_planning_metrics(episodes, trajectories, use_teacher_dst=True)
        NE, SR, OSR, SPL = metrics.mean_final_pos_to_goal_dist, metrics.success_rate_final_pos_to_goal, metrics.success_rate_oracle_pos_to_goal, metrics.success_rate_weighted_by_path_length
        result = {
            "type": eval_type[i],
            "metrics": {
                "NE": NE,
                "SR": SR,
                "OSR": OSR,
                "SPL": SPL
            }
        }
        total_result.append(result)
        with open(result_save_path1, "w") as f:
            json.dump(total_result, f, indent=4)
        with open(actions_save_path1, "w") as f:
            json.dump(total_actions, f, indent=4)
        print(f"Eval Metrics : ", metrics)
        print(f'NE:{NE}\nSR:{SR}\nOSR:{OSR}\nSPL:{SPL}')
        print(f"================= End eval {eval_type[i]} =================\n\n")
    
    return 

def test():
    eval_data_paths = ["./data/AirNav/test/info_test_easy.json",
                       "./data/AirNav/test/info_test_medium.json",
                       "./data/AirNav/test/info_test_hard.json"]
    airnav_data_paths = ["./data/AirNav/test/airnav_test_easy.json",
                          "./data/AirNav/test/airnav_test_medium.json",
                          "./data/AirNav/test/airnav_test_hard.json"]
    eval_type = ["easy","medium","hard"]
    total_result = []
    
    total_trajectories = dict()
    total_episodes = dict()
    total_actions = dict()
    for i in range(3):
        
        print(f"================= Start eval {eval_type[i]} =================")
        airnav_data_path = airnav_data_paths[i]
        
        airnav_data = AirNavData(airnav_data_path)
        airnav_data_dict = dict()
        
        airnav_index = dict()
        for k, item in enumerate(airnav_data):
            episode_id = item.episode.id[-1]
            airnav_index[episode_id] = k

        eval_data_path = eval_data_paths[i]
        
        with open(eval_data_path, 'r') as f:
            eval_data = json.load(f)

        episodes, trajectories, actions = eval_test(airnav_data, airnav_index, eval_data, max_workers=max_workers)
        
        total_actions[eval_type[i]] = actions
        total_trajectories.update(trajectories)
        total_episodes.update(episodes)
        
        metrics = eval_planning_metrics(episodes, trajectories, use_teacher_dst=True)
        NE, SR, OSR, SPL = metrics.mean_final_pos_to_goal_dist, metrics.success_rate_final_pos_to_goal, metrics.success_rate_oracle_pos_to_goal, metrics.success_rate_weighted_by_path_length
        result = {
            "type": eval_type[i],
            "metrics": {
                "NE": NE,
                "SR": SR,
                "OSR": OSR,
                "SPL": SPL
            }
        }
        total_result.append(result)
        with open(result_save_path2, "w") as f:
            json.dump(total_result, f, indent=4)
        with open(actions_save_path2, "w") as f:
            json.dump(total_actions, f, indent=4)
        print(f"Eval Metrics : ", metrics)
        print(f'NE:{NE}\nSR:{SR}\nOSR:{OSR}\nSPL:{SPL}')
        print(f"================= End eval {eval_type[i]} =================\n\n")
    
    metrics = eval_planning_metrics(total_episodes, total_trajectories, use_teacher_dst=True)
    NE, SR, OSR, SPL = metrics.mean_final_pos_to_goal_dist, metrics.success_rate_final_pos_to_goal, metrics.success_rate_oracle_pos_to_goal, metrics.success_rate_weighted_by_path_length
    print("Final Eval Metrics: ", metrics)
    print(f'NE:{NE}\nSR:{SR}\nOSR:{OSR}\nSPL:{SPL}')
    result = {
        "type": "all",
        "metrics": {
            "NE": NE,
            "SR": SR,
            "OSR": OSR,
            "SPL": SPL
        }
    }
    total_result.append(result)
    with open(result_save_path2, "w") as f:
        json.dump(total_result, f, indent=4)
    return 

if __name__ == "__main__":
    val()
    test()