import argparse
import os
import datasets
import json
from PIL import Image
from verl.utils.hdfs_io import copy, makedirs
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import re
from pathlib import Path
import rasterio
def get_prompt(data):
    
    history_views = data["history_views"]
    history_views_path = []
    indices = [-7, -4, -2, -1]
    for index in indices:
        if len(history_views) >= abs(index):  # 确保索引存在，防止越界
            history_views_path.append(history_views[index])
    
    cur_view_path = data["cur_view"]
    cur_pose = data["cur_position"]
    instruction = data["instruction"]
    #target_description = data["target_description"]
    history_actions = json.dumps(data["history_actions"])
    future_actions = json.dumps(data["future_actions"])

    sft_data = dict()
    # 所有图片信息
    images = []
    images.extend(history_views_path)
    images.append(cur_view_path)


    # 文本信息
    messages = []
    system_content = dict()
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
    system_content["role"] = "system"
    system_content["content"] = system_prompt
    messages.append(system_content)

    user_content = dict()
    user_content["role"] = "user"
    user_prompt = ""
    for i in range(len(history_views_path)):
        user_prompt += "<image>\n"
        user_prompt += f"(Above is historical view {i + 1})\n"
    
    user_prompt += "<image>\n"
    user_prompt += "(Above is the current view)\n"
    user_prompt += f"""
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
    user_content["content"] = user_prompt
    messages.append(user_content)

    assistant_content = dict()
    assistant_content["role"] = "assistant"
    assistant_content["content"] = future_actions
    messages.append(assistant_content)

    sft_data["messages"] = messages
    sft_data["images"] = images

    return system_prompt, user_prompt, images

# image_dir = Path("/mnt/vepfs/ryj/UAV_benchmark/data/rgbd-new")
# raster_cache = {
#     raster_path.stem: rasterio.open(raster_path)
#     for raster_path in image_dir.glob("*.tif")
# }

def get_map_name(episode_id):
    parts = episode_id.split('_')
    map_name = parts[0] + '_' + parts[1] + '_' + parts[2]
    return map_name

def trans_data(benchmark_data):
    trans_list = []
    for example in benchmark_data:
        data = dict()

        data["ground_truth"] = json.dumps(example["future_actions"])

        system_prompt, user_prompt, images = get_prompt(example)
        data["prompt"] = [{"role": "system","content":system_prompt}
                          ,{"role": "user","content":user_prompt}]
        
        #image_path = example["images"][0]
        #data["images"] = [{'image':image_path}]
        #data["images"] = [{'image':img_path} for img_path in images]
        # 符合from qwen_vl_utils import fetch_image 格式要求
        image_format = []
        for image in images:
            image_data = dict()
            image_data["image"] = image
            image_format.append(image_data)
        data["images"] = image_format

        data["target_px"] = example["final_target_px"]
        data["map_name"] = get_map_name(example["episode_id"])
        data["cur_px"] = example["cur_position_px"]
        data["cur_pose"] = example["cur_position"]
        #data["raster"] = raster_cache[data["map_name"]]


        trans_list.append(data)
    return trans_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/path/to/train/data/train.json"

    with open(data_source, 'r') as f:
            benchmark_data = json.load(f)
    benchmark_trans = trans_data(benchmark_data)

    print(len(benchmark_trans))
    train_dataset = HFDataset.from_list(benchmark_trans)

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            ground_truth = example.pop("ground_truth")
            images = example.pop("images")
            target_px = example.pop("target_px")
            map_name = example.pop("map_name")
            cur_px = example.pop("cur_px")
            cur_pose = example.pop("cur_pose")
            #raster = example.pop("raster")
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "images": images,
                "ability": "navigation",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                # "target": [{'content':f"<think>think here</think>\\boxed{answer}",
                #             "role": "assistant"}],
                # "target": [{'content':f"{target}",
                #             "role": "assistant"}],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "target_px": target_px,
                    "map_name": map_name,
                    "cur_px": cur_px,
                    "cur_pose": cur_pose
                },
            }
            return data
        
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=32)
    print("train_dataset", len(train_dataset))
    print(train_dataset[0]["prompt"])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, "AirNav_rl.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
