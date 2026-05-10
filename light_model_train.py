from transformers import BertTokenizerFast
from gsamllavanav.models.cma import CMA
from gsamllavanav.models.seq2seq import Seq2Seq
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

BLACK_IMG_PATH = "./gsamllavanav/models/black_view.jpg"
HISTORY_INDICES = [-7, -4, -2, -1]
PRED_HORIZON = 8
ACTION_TO_ID = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
}

if not os.path.exists(BLACK_IMG_PATH):
    os.makedirs(os.path.dirname(BLACK_IMG_PATH), exist_ok=True)
    black_img = Image.new("RGB", (448, 448), (0, 0, 0))
    black_img.save(BLACK_IMG_PATH)


class AirNavDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform
        local_path = "./model_weight/bert-base-uncased"
        model_name = "bert-base-uncased"
        if os.path.exists(local_path):
            self.tokenizer = BertTokenizerFast.from_pretrained(local_path)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            self.tokenizer.save_pretrained(local_path)

    def __len__(self):
        return len(self.data)

    def _select_history_views(self, history_views):
        selected = []
        for index in HISTORY_INDICES:
            if len(history_views) >= abs(index):
                selected.append(history_views[index])
            else:
                selected.append(BLACK_IMG_PATH)
        return selected

    def _build_label(self, item):
        if "label" in item:
            return torch.tensor(item["label"], dtype=torch.long)

        future_actions = item.get("future_actions", [])
        label = []
        for action in future_actions:
            action_key = str(action).upper()
            label.append(ACTION_TO_ID.get(action_key, ACTION_TO_ID["STOP"]))

        if len(label) < PRED_HORIZON:
            label.extend([ACTION_TO_ID["STOP"]] * (PRED_HORIZON - len(label)))
        else:
            label = label[:PRED_HORIZON]

        return torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]

        instruction = item["instruction"]
        tokenized_instruction = self.tokenizer(
            instruction,
            padding="max_length",
            truncation=True,
            max_length=1000,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"].squeeze(0)

        actions = item.get("history_actions", [])
        if isinstance(actions, list):
            actions = str(actions)
        if actions == "[]":
            actions = "no action"

        tokenized_actions = self.tokenizer(
            actions,
            padding="max_length",
            truncation=True,
            max_length=1000,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"].squeeze(0)

        cur_img = Image.open(item["cur_view"]).convert("RGB")
        history_views = item.get("history_views", [])
        if not isinstance(history_views, list):
            history_views = []
        selected_history_views = self._select_history_views(history_views)
        his_img1 = Image.open(selected_history_views[0]).convert("RGB")
        his_img2 = Image.open(selected_history_views[1]).convert("RGB")
        his_img3 = Image.open(selected_history_views[2]).convert("RGB")
        his_img4 = Image.open(selected_history_views[3]).convert("RGB")

        if self.transform:
            cur_img = self.transform(cur_img)
            his_img4 = self.transform(his_img4)
            his_img3 = self.transform(his_img3)
            his_img2 = self.transform(his_img2)
            his_img1 = self.transform(his_img1)

        label = self._build_label(item)

        return {
            "input_ids": tokenized_instruction,
            "history_actions": tokenized_actions,
            "cur_rgb": cur_img,
            "his_rgb_4": his_img4,
            "his_rgb_3": his_img3,
            "his_rgb_2": his_img2,
            "his_rgb_1": his_img1,
            "label": label,
        }


def load_model(model, optimizer=None, path="checkpoints/model.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    print(f"✅ 模型已从 {path} 加载，继续训练从 epoch {epoch + 1} 开始")
    return model


def save_model(model, optimizer, epoch, loss, path="./model_weight", filename="model.pth"):
    if model_type == "CMA":
        path = os.path.join(path, "CMA")
    else:
        path = os.path.join(path, "Seq2Seq")

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_path,
    )

    print(f"✅ 模型已保存到 {save_path}")


def train(model, dataloader, optimizer, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            history_actions = batch["history_actions"].to(device)
            cur_rgb = batch["cur_rgb"].to(device)
            his_rgb_4 = batch["his_rgb_4"].to(device)
            his_rgb_3 = batch["his_rgb_3"].to(device)
            his_rgb_2 = batch["his_rgb_2"].to(device)
            his_rgb_1 = batch["his_rgb_1"].to(device)
            labels = batch["label"].to(device)

            batch_size = input_ids.size(0)
            rnn_states = model.get_initial_recurrent_hidden_states(batch_size, device)
            not_done_masks = torch.ones(batch_size, dtype=bool, device=device)

            pred_actions = model(
                input_ids,
                history_actions,
                cur_rgb,
                his_rgb_4,
                his_rgb_3,
                his_rgb_2,
                his_rgb_1,
                rnn_states,
                not_done_masks,
            )
            pred_actions = pred_actions.view(batch_size, 8, 4)

            loss = criterion(pred_actions.transpose(1, 2), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            save_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                filename=f"model_{epoch}.pth",
            )

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def train_main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_path = "./data/AirNav/train/your_path_to_train"
    dataset = AirNavDataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=8)

    if model_type == "CMA":
        model = CMA()
    else:
        model = Seq2Seq()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train(model, dataloader, optimizer, device, num_epochs=20)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_type = "CMA"  # "CMA" or "Seq2Seq"


if __name__ == "__main__":
    train_main()
