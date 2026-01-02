from transformers import BertTokenizerFast
from gsamllavanav.models.cma import CMA
from gsamllavanav.models.seq2seq import Seq2Seq
from navgym.models.AirNavData import AirNavData
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from navgym.models.NavGym import NavGym
from datetime import datetime
from navgym.tools.EvalTools import eval_planning_metrics
import matplotlib.pyplot as plt


def load_model(model, optimizer=None, path="checkpoints/model.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    
    return model


black_img_path = "./gsamllavanav/models/black_view.jpg"

if not os.path.exists(black_img_path):
    black_img = Image.new("RGB", (448, 448), (0, 0, 0)) 
    black_img.save(black_img_path)
    
def eval_one_episode(airnav_data, airnav_index, eval_data, key, model, tokenizer, transform):
    
    single_eval_data = eval_data[key]
    cur_episode_id = eval_data[key]["episode_id"]
    navGym = NavGym(airnav_data[airnav_index[cur_episode_id]], data_dir=os.path.abspath('./EvalPhotoData'))
    history_views = []
    for _ in range(7):
        history_views.append("black_img_path")
        
    history_actions = []
    cur_pose = navGym.start_pose
    cur_trajectory = [cur_pose]
    total_steps = 0
    total_actions = []
    k = 0
    time = datetime.now().strftime("%Y%m%d%H%M%S%f")

    while True:

        instruction = single_eval_data["instruction"]
        history_actions_str = str(history_actions)
        if history_actions_str == "[]":
            history_actions_str = "no action" 

        tokenized_instruction = tokenizer(
            instruction,
            padding="max_length",   
            truncation=True,        
            max_length=1000,          
            return_tensors="pt",     
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"].to(device)

        tokenized_actions = tokenizer(
            history_actions_str,
            padding="max_length",   
            truncation=True,        
            max_length=1000,          
            return_tensors="pt",     
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"].to(device)

        save_path = navGym.father_image_dir + f"/{time}_case_{k}"
        os.makedirs(save_path, exist_ok=True)
        
        his_img4 = Image.open(history_views[-1]).convert("RGB")
        his_img3 = Image.open(history_views[-2]).convert("RGB")
        his_img2 = Image.open(history_views[-4]).convert("RGB")
        his_img1 = Image.open(history_views[-7]).convert("RGB")
        
        his_img4 = transform(his_img4).unsqueeze(0).to(device)
        his_img3 = transform(his_img3).unsqueeze(0).to(device)
        his_img2 = transform(his_img2).unsqueeze(0).to(device)
        his_img1 = transform(his_img1).unsqueeze(0).to(device)

        _, cur_view = navGym._get_cur_drone_view()
        plt.imsave(save_path + "/cur_view.jpg", cur_view)
        
        cur_img = Image.open(save_path + "/cur_view.jpg").convert("RGB")
        cur_img = transform(cur_img).unsqueeze(0).to(device)

        history_views.append(save_path + "/cur_view.jpg")

        B = 1
        rnn_states = model.get_initial_recurrent_hidden_states(B, device) 
        not_done_masks = torch.ones(B, dtype=bool, device=device)

        pred_actions = model(
                tokenized_instruction,
                tokenized_actions,
                cur_img,
                his_img4,
                his_img3,
                his_img2,
                his_img1,
                rnn_states,
                not_done_masks,
            )
        
        pred_actions = pred_actions.view(B, 8, 4)  # [B,8,4]
        pred_action_ids = pred_actions.argmax(dim=-1)  # [B, 8]

        #pred_action_ids = pred_action_ids[0] # 减去batch那一维

        action_str_dict = {
            1:"MOVE_FORWARD", 
            3:"TURN_LEFT",
            2:"TURN_RIGHT",
            0:"STOP"
        }

        actions_list = pred_action_ids[0].tolist()
        action_str = []
        for id in actions_list:
            action_str.append(action_str_dict[id])

        history_actions.extend(action_str)
        

        for action in actions_list:
            if action == 0:
                break
            cur_trajectory.append(navGym.cur_pose)
            navGym.step(action, savefig=False, saveviewfig=False)

        total_steps += len(actions_list)
        total_actions.extend(action_str)

        #_, map_with_tar = navGym._get_cur_trajectory_map()
        
        #plt.imsave(save_path + "/map_with_tra.jpg", map_with_tar)

        cur_px = navGym.cur_position_px
        
        #out_of_map = not (0 <= cur_px[0] < map_size[0] and 0 <= cur_px[1] < map_size[1])

        if (0 in actions_list) or total_steps >= 160:  
            break
        k += 1

    result_data = {
        "actions": total_actions,
        "end_position": navGym.cur_position,
        "end_position_px": navGym.cur_position_px,
        "total_steps": total_steps
    }
    return (key, airnav_data[airnav_index[cur_episode_id]].episode, cur_trajectory, result_data)

def eval(airnav_data, airnav_index, eval_data, max_workers=20):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    if model_type == "CMA":
        model = CMA()
        model = load_model(model = model,path = f"./model_weight/CMA.pth",device = device)
    else:
        model = Seq2Seq()
        model = load_model(model = model,path = f"./model_weight/Seq2Seq.pth",device = device)
        

    model.to(device)
    model.eval()

    local_path = "./model_weight/bert-base-uncased"
    model_name = "bert-base-uncased"

    if os.path.exists(local_path):
        tokenizer = BertTokenizerFast.from_pretrained(local_path)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        tokenizer.save_pretrained(local_path) 
    
    trajectories = dict()
    episodes = dict()
    action_records = dict()
    total = len(eval_data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(eval_one_episode, airnav_data, airnav_index, eval_data, key, model, tokenizer, transform)
            for key in eval_data.keys()
        ]
        with tqdm(total=total) as pbar:
            for future in as_completed(futures):
                key, eid, traj, result_data = future.result()
                if eid is not None:
                    episodes[key] = eid
                    trajectories[key] = traj
                    action_records[key] = result_data
                pbar.update(1)

    return episodes, trajectories, action_records


def val():
    eval_data_paths = ["./data/AirNav/val/info_val_seen.json",
                       "./data/AirNav/val/info_val_unseen.json"]
    airnav_data_paths = ["./data/AirNav/val/airnav_val_seen.json",
                          "./data/AirNav/val/airnav_val_unseen.json"]
    eval_type = ["val_seen","val_unseen"]
    
    result_save_path = f"./result/val/{model_type}.json"
    actions_save_path = f"./result/val/{model_type}.json"

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

        episodes, trajectories, actions = eval(airnav_data, airnav_index, eval_data, max_workers=20)
        
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
        with open(result_save_path, "w") as f:
            json.dump(total_result, f, indent=4)
        with open(actions_save_path, "w") as f:
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
    
    result_save_path = f"./result/test/{model_type}.json"
    actions_save_path = f"./result/test/{model_type}.json"

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

        episodes, trajectories, actions = eval(airnav_data, airnav_index, eval_data, max_workers=20)
        
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
        with open(result_save_path, "w") as f:
            json.dump(total_result, f, indent=4)
        with open(actions_save_path, "w") as f:
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
    with open(result_save_path, "w") as f:
        json.dump(total_result, f, indent=4)
    return 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_type="CMA"

if __name__ == "__main__":
    val()
    test()