"""
Convert train_data_generate.py output (train.json) to LLaMA-Factory sharegpt format.

Mirrors eval.py's prompt template exactly so train/inference distributions align:
  - System prompt: identical to eval.py
  - User content: <image> placeholders interleaved with view-index captions, then text block
  - History view sampling: indices [-7, -4, -2, -1] from full history (same as eval.py:239)
  - Assistant: json.dumps(future_actions)

Usage:
  python train_to_sharegpt.py \
      --input  data/AirNav/train/train.json \
      --output LLaMA-Factory/data/airnav_sft.json
"""
import argparse
import json
import os
from pathlib import Path

SYSTEM_PROMPT = """
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

USER_TEXT_TEMPLATE = """
## Text Input
- **Navigation instruction**: {instruction}
- **Current state of the UAV**: {cur_pose} (x, y, z in meters; heading in degrees)
- **Previously executed actions**: {history_actions}
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

HISTORY_INDICES = [-7, -4, -2, -1]


def sample_history(history_views):
    sampled = []
    for idx in HISTORY_INDICES:
        if len(history_views) >= abs(idx):
            sampled.append(history_views[idx])
    return sampled


def build_user_content(history_paths, cur_path, instruction, cur_pose, history_actions):
    parts = []
    for i, _ in enumerate(history_paths, 1):
        parts.append(f"<image>\n(Above is historical view {i})")
    parts.append("<image>\n(Above is the current view)")

    text = USER_TEXT_TEMPLATE.format(
        instruction=instruction,
        cur_pose=str(cur_pose),
        history_actions=(
            json.dumps(history_actions) if history_actions else "None"
        ),
    )
    parts.append(text)
    return "\n".join(parts)


def convert_one(case, image_path_mode):
    history_paths = sample_history(case["history_views"])
    cur_path = case["cur_view"]
    all_image_paths = history_paths + [cur_path]

    if image_path_mode == "absolute":
        all_image_paths = [os.path.abspath(p) for p in all_image_paths]
    # "as-is" leaves whatever paths train_data_generate.py wrote

    user_content = build_user_content(
        history_paths=history_paths,
        cur_path=cur_path,
        instruction=case["instruction"],
        cur_pose=case["cur_position"],
        history_actions=case["history_actions"],
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(case["future_actions"])},
        ],
        "images": all_image_paths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to train.json from train_data_generate.py")
    ap.add_argument("--output", required=True, help="output path for LLaMA-Factory sharegpt JSON")
    ap.add_argument(
        "--image-paths",
        choices=["absolute", "as-is"],
        default="absolute",
        help="absolute: rewrite image paths to absolute (recommended). as-is: keep what train.json contains.",
    )
    ap.add_argument("--limit", type=int, default=0, help="optional cap on cases (0 = all)")
    args = ap.parse_args()

    with open(args.input) as f:
        cases = json.load(f)
    if args.limit > 0:
        cases = cases[: args.limit]

    converted = [convert_one(c, args.image_paths) for c in cases]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    # sanity check on first sample
    sample = converted[0]
    n_image_tags = sample["messages"][1]["content"].count("<image>")
    n_images = len(sample["images"])
    assert n_image_tags == n_images, f"<image> count {n_image_tags} != images count {n_images}"

    print(f"converted {len(converted)} cases -> {args.output}")
    print(f"first sample: {n_images} images, "
          f"history={len(sample_history(cases[0]['history_views']))}, "
          f"future_actions={cases[0]['future_actions']}")


if __name__ == "__main__":
    main()
