import argparse
import ast
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import torch
from itertools import product
from transformers import CLIPImageProcessor
from uninavid.model.builder import load_pretrained_model
from uninavid.mm_utils import get_model_name_from_path
from uninavid.train.train import preprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--model-path",
        default="./checkpoints/uninavid",
        help="Path to the trained Uni-NaVid checkpoint.",
    )
    return parser.parse_args()


ARGS = parse_args()
MODEL_PATH = ARGS.model_path
model_name = get_model_name_from_path(MODEL_PATH)

tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, None, model_name)
model.eval()

processor_path = "./uninavid/processor/clip-patch14-224"
processor = CLIPImageProcessor.from_pretrained(processor_path)

ACTION_SPACE = ("STOP", "MOVE_FORWARD", "TURN_RIGHT", "TURN_LEFT")
ACTION_LIST_LEN = 8
TRIE_END = "__end__"
infer_lock = threading.Lock()


def format_action_list(actions):
    return "[" + ", ".join(f"'{action}'" for action in actions) + "]"


def build_action_trie(tokenizer, action_space, list_len):
    trie = {}
    max_tokens = 0
    for actions in product(action_space, repeat=list_len):
        token_ids = tokenizer.encode(format_action_list(actions), add_special_tokens=False)
        max_tokens = max(max_tokens, len(token_ids))
        node = trie
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        node[TRIE_END] = True
    return trie, max_tokens


def make_prefix_allowed_tokens_fn(prompt_len, trie, eos_token_id):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        generated_ids = input_ids[prompt_len:].tolist()
        node = trie
        for token_id in generated_ids:
            if token_id not in node:
                return [eos_token_id]
            node = node[token_id]

        allowed = [token_id for token_id in node.keys() if token_id != TRIE_END]
        if TRIE_END in node:
            allowed.append(eos_token_id)
        return allowed or [eos_token_id]

    return prefix_allowed_tokens_fn


ACTION_TRIE, ACTION_MAX_NEW_TOKENS = build_action_trie(tokenizer, ACTION_SPACE, ACTION_LIST_LEN)

sample = {
    "image": "examples/uav_sample/cur_view.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n## Role\nYou are an expert navigation assistant for a UAV (Unmanned Aerial Vehicle) flight simulator.\n\n## Task Objective\nThe UAV operates in a 3D urban environment with visible roads, buildings, and landmarks.\nYour task is to predict the next sequence of UAV actions based on:\n1. A given natural language navigation instruction,\n2. The current state of the UAV, including its position and heading angle,\n3. The current top-down UAV view image,\n4. Up to four historical top-down view images from previous time steps (if available),\n5. The previously executed UAV actions (if available).\n\n## Text Input\n- **Navigation instruction**: Move forward and keep going straight, watching for a small light‑colored rectangular outbuilding tucked into a narrow garden plot beside a row of houses. Continue moving forward directly toward that little outbuilding, and when you arrive beside it, stop.\n- **Current state of the UAV**: [330.328723607643, 461.34559246293816, 60.02721669169363, -149.79479838175214] (x, y, z in meters; heading in degrees)\n\n## Image Input\n**UAV (Unmanned Aerial Vehicle) Top-Down View Sequence**\n- Historical top-down views (from oldest to newest) show the UAV’s past observations.\n- The last image is the current top-down view of the UAV.\n- In all images, the **top of the image corresponds to the UAV’s forward direction** (its heading).\n\nBased on the navigation instruction, the UAV’s current state, the previously executed actions (which can help infer the UAV’s current orientation and progress), and the provided images, predict how the UAV should move **step by step** to follow the instruction accurately.\n\n## Prediction Rules\n1. Predict no more than **8 future actions** for the UAV to execute.\n2. If the target location is reachable in fewer than 8 actions, output less than 8 actions sequence and end with **\"STOP\"**. Otherwise, it clearly requires more than 8 actions to approach the target, output exactly 8 future actions.\n3. You **must** output **\"STOP\"** if the UAV has already reached the described target.\n\n## Discrete Action Space\n- `STOP`: stop the flight\n- `MOVE_FORWARD`: move straight 5 meters in the current heading\n- `TURN_RIGHT`: rotate right 30°\n- `TURN_LEFT`: rotate left 30°"
        },
        {
            "from": "gpt",
            "value": "['MOVE_FORWARD', 'MOVE_FORWARD', 'MOVE_FORWARD', 'MOVE_FORWARD', 'MOVE_FORWARD', 'STOP']"
        }
    ]
}

prompt = sample["conversations"][0]["value"]
image = sample["image"]


app = FastAPI()


def generate(prompt, image):
    source = [{"from": "human", "value": prompt}]
    data_dict = preprocess(
        [source],
        tokenizer,
        has_image=True
    )

    image = Image.open(image).convert("RGB")
    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(dtype=model.dtype)

    data_dict = dict(
        input_ids=data_dict["input_ids"][0].unsqueeze(0).to(model.device),
        images=[image.to(model.device)],
        prompts=[prompt],
    )
    # `update_prompt()` mutates model state, so one process should serialize
    # requests that share the same model instance.
    with infer_lock:
        with torch.inference_mode():
            model.update_prompt([[prompt]])
            prompt_len = data_dict["input_ids"].shape[1]
            output_ids = model.generate(
                **data_dict,
                do_sample=False,
                max_new_tokens=ACTION_MAX_NEW_TOKENS + 1,
                use_cache=True,
                prefix_allowed_tokens_fn=make_prefix_allowed_tokens_fn(
                    prompt_len=prompt_len,
                    trie=ACTION_TRIE,
                    eos_token_id=tokenizer.eos_token_id,
                ),
            )
    output = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return ast.literal_eval(output)

from pydantic import BaseModel

class Item(BaseModel):
    prompt: str
    image: str

@app.post("/generate")
async def generate_endpoint(item: Item):
    return await run_in_threadpool(generate, item.prompt, item.image)

if __name__ == "__main__":
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)