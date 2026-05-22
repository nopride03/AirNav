import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# gsamllavanav uses relative paths like `data/cityrefer/objects.json` and
# `navgym` is a top-level package, so we both add the AirNav root to sys.path
# AND chdir into it before importing.
_AIRNAV_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = Path(__file__).resolve().parent
if str(_AIRNAV_ROOT) not in sys.path:
    sys.path.insert(0, str(_AIRNAV_ROOT))
os.chdir(_AIRNAV_ROOT)

from navgym.models.NavGym import NavGym
from navgym.models.AirNavData import AirNavData
from navgym.tools.EvalTools import eval_planning_metrics

UNIX_TIMESTAMP = int(time.time())
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MAX_STEPS = 160

# Configurable paths via environment variables. Defaults are relative to the
# AirNav root directory (we chdir'd there above).
data_dir = Path(os.environ.get("DATA_DIR", "data/AirNav"))
image_dir = os.environ.get("IMAGE_DIR", "data/rgbd-new")
eval_data_paths = [
    data_dir / "val" / "info_val_seen.json",
    data_dir / "val" / "info_val_unseen.json",
    data_dir / "test" / "info_test_easy.json",
    data_dir / "test" / "info_test_medium.json",
    data_dir / "test" / "info_test_hard.json",
]
airnav_data_paths = [
    data_dir / "val" / "airnav_val_seen.json",
    data_dir / "val" / "airnav_val_unseen.json",
    data_dir / "test" / "airnav_test_easy.json",
    data_dir / "test" / "airnav_test_medium.json",
    data_dir / "test" / "airnav_test_hard.json",
]
eval_types = [
    "val_seen",
    "val_unseen",
    "test_easy",
    "test_medium",
    "test_hard",
]

# Write results back under the Uni-NaVid baseline directory.
save_dir = _BASELINE_DIR / "results" / str(UNIX_TIMESTAMP)
save_dir.mkdir(parents=True, exist_ok=True)

act2id = {"MOVE_FORWARD": 1, "TURN_LEFT": 3, "TURN_RIGHT": 2, "STOP": 0}

TEMPLATE = """
<image>
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

## Text Input
- **Navigation instruction**: {instruction}
- **Current state of the UAV**: {cur_pose} (x, y, z in meters; heading in degrees)

## Image Input
**UAV (Unmanned Aerial Vehicle) Top-Down View Sequence**
- Historical top-down views (from oldest to newest) show the UAV's past observations.
- The last image is the current top-down view of the UAV.
- In all images, the **top of the image corresponds to the UAV's forward direction** (its heading).

Based on the navigation instruction, the UAV's current state, the previously executed actions (which can help infer the UAV's current orientation and progress), and the provided images, predict how the UAV should move **step by step** to follow the instruction accurately.

## Prediction Rules
1. Predict no more than **8 future actions** for the UAV to execute.
2. If the target location is reachable in fewer than 8 actions, output less than 8 actions sequence and end with **"STOP"**. \
Otherwise, it clearly requires more than 8 actions to approach the target, output exactly 8 future actions.
3. You **must** output **"STOP"** if the UAV has already reached the described target.

## Discrete Action Space
- `STOP`: stop the flight
- `MOVE_FORWARD`: move straight 5 meters in the current heading
- `TURN_RIGHT`: rotate right 30°
- `TURN_LEFT`: rotate left 30°
""".strip()


def get_actions(data_dict):
    prompt = TEMPLATE.format(
        instruction=data_dict["instruction"],
        cur_pose=data_dict["cur_position"],
    )
    url = os.environ.get("UNINAVID_URL", "[REDACTED_URL]") + "/generate"
    data = {
        "prompt": prompt,
        "image": data_dict["cur_view"],
    }
    try:
        response = requests.post(url, json=data).json()
        return response
    except:
        return get_random_actions()


def get_random_actions():
    weights = [5, 2, 2, 1]
    actions_list = list(act2id.keys())
    actions = []
    for _ in range(8):
        action = random.choices(actions_list, weights)[0]
        actions.append(action)
        if action == "STOP":
            break
    return actions


def eval_one_episode(airnav_data, airnav_index, eval_data, key):
    cur_episode_id = eval_data[key]["episode_id"]
    if cur_episode_id not in airnav_index:
        print(f"Warning: {cur_episode_id} not in airnav_data")
        return None, None, None, None

    navGym = NavGym(
        airnav_data[airnav_index[cur_episode_id]],
        data_dir=os.path.abspath("./EvalPhotoData"),
    )
    history_views = []
    history_actions = []
    cur_pose = navGym.start_pose
    cur_trajectory = [cur_pose]
    total_steps = 0
    total_actions = []
    k = 0
    time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    save_path = navGym.father_image_dir + f"/{time}_case_{k}"
    os.makedirs(save_path, exist_ok=True)
    while True:
        _, cur_view = navGym._get_cur_drone_view()
        cur_view_path = save_path + "/cur_view.jpg"
        plt.imsave(cur_view_path, cur_view)
        data_dict = {
            "episode_id_case": navGym.episode_id + f"_case{k}",
            "instruction": eval_data[key]["instruction"],
            "history_actions": history_actions.copy(),
            "history_views": history_views.copy(),
            "cur_view": cur_view_path,
            "cur_position": navGym.cur_position,
        }
        actions = get_actions(data_dict)
        action_ids = []
        for act in actions:
            act_id = act2id[act]
            action_ids.append(act_id)
            total_actions.append(act)
            if act_id == 0:
                break
            navGym.step(act_id, savefig=False, saveviewfig=False)
            total_steps += 1
            cur_trajectory.append(navGym.cur_pose)
        if action_ids[-1] == 0 or total_steps >= MAX_STEPS:
            break
        history_actions.extend(actions)
        history_views.append(cur_view_path)
        k += 1

    result_data = {
        "actions": total_actions,
        "end_position": navGym.cur_position,
        "end_position_px": navGym.cur_position_px,
        "total_steps": total_steps,
    }
    return (
        key,
        airnav_data[airnav_index[cur_episode_id]].episode,
        cur_trajectory,
        result_data,
    )


def eval_thread_pool(airnav_data, airnav_index, eval_data, max_workers=MAX_WORKERS):
    trajectories, episodes, action_records = {}, {}, {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                eval_one_episode, airnav_data, airnav_index, eval_data, key
            )
            for key in eval_data.keys()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Eval"):
            key, eid, traj, actions = future.result()
            if eid is not None:
                episodes[key] = eid
                trajectories[key] = traj
                action_records[key] = actions
    return episodes, trajectories, action_records


def evaluate(eval_data_path, airnav_data_path, eval_type):
    print(f"================= Start Eval <{eval_type}> =================")
    airnav_data = AirNavData(airnav_data_path, image_dir=image_dir)
    airnav_index = dict()
    for k, item in enumerate(airnav_data):
        episode_id = item.episode.id[-1]
        airnav_index[episode_id] = k
    with open(eval_data_path, "r") as f:
        eval_data = json.load(f)
    episodes, trajectories, actions = eval_thread_pool(
        airnav_data, airnav_index, eval_data
    )

    metrics = eval_planning_metrics(episodes, trajectories, use_teacher_dst=True)
    result = {
        "type": eval_type,
        "metrics": {
            "NE": metrics.mean_final_pos_to_goal_dist,
            "ONE": metrics.mean_oracle_pos_to_goal_dist,
            "SR": metrics.success_rate_final_pos_to_goal,
            "OSR": metrics.success_rate_oracle_pos_to_goal,
            "SPL": metrics.success_rate_weighted_by_path_length,
        },
    }
    with open(save_dir / f"{eval_type}_result.json", "w") as f:
        json.dump(result, f, indent=4)
    with open(save_dir / f"{eval_type}_actions.json", "w") as f:
        json.dump(actions, f, indent=4)

    for k, v in result["metrics"].items():
        print(f"{k: <3}: {v:.4f}")
    print(f"================= End Eval <{eval_type}> =================")
    return episodes, trajectories


def eval_test_average(test_episodes_all, test_trajectories_all, case_counts):
    print("================= Start Eval <test_average> =================")
    metrics = eval_planning_metrics(
        test_episodes_all, test_trajectories_all, use_teacher_dst=True
    )
    result = {
        "type": "test_average",
        "metrics": {
            "NE": metrics.mean_final_pos_to_goal_dist,
            "ONE": metrics.mean_oracle_pos_to_goal_dist,
            "SR": metrics.success_rate_final_pos_to_goal,
            "OSR": metrics.success_rate_oracle_pos_to_goal,
            "SPL": metrics.success_rate_weighted_by_path_length,
        },
        "case_counts": case_counts,
        "total_cases": len(test_episodes_all),
    }
    with open(save_dir / "test_average_result.json", "w") as f:
        json.dump(result, f, indent=4)
    for k, v in result["metrics"].items():
        print(f"{k: <3}: {v:.4f}")
    print("================= End Eval <test_average> =================")


if __name__ == "__main__":
    print("Uni-NaVid Evaluation")
    print(UNIX_TIMESTAMP)
    test_episodes_all = {}
    test_trajectories_all = {}
    test_case_counts = {}
    for eval_data_path, airnav_data_path, eval_type in zip(
        eval_data_paths, airnav_data_paths, eval_types
    ):
        episodes, trajectories = evaluate(
            eval_data_path, airnav_data_path, eval_type
        )
        if eval_type.startswith("test_"):
            test_case_counts[eval_type] = len(episodes)
            for k, v in episodes.items():
                test_episodes_all[f"{eval_type}/{k}"] = v
            for k, v in trajectories.items():
                test_trajectories_all[f"{eval_type}/{k}"] = v
    if test_episodes_all:
        eval_test_average(test_episodes_all, test_trajectories_all, test_case_counts)
    print("Uni-NaVid Evaluation")
    print(UNIX_TIMESTAMP)
