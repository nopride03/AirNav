"""Pi0 VLN inference server (one backend per GPU).

Mirrors Uni-NaVid's ``api.py`` contract so that the same gateway / client
harness used for the other baselines can talk to pi0 too:

    POST /generate
        body: {"prompt": "<full human-turn text>", "image": "/abs/path.jpg"}
        returns: ["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]

The returned list is variable-length: we decode the full ``action_horizon=8``
chunk produced by pi0, then cut at the first ``STOP`` (inclusive), matching
the convention used by the training GT. If no ``STOP`` is predicted, the full
8-element list is returned.

Env knobs
---------
``OPENPI_CONFIG``       TrainConfig name (default ``pi0_vln_lora``).
``OPENPI_CHECKPOINT``   Path to the step directory, e.g.
                        ``checkpoints/pi0_vln_lora/<exp>/<step>``. If not
                        set, auto-picks the latest step under
                        ``checkpoints/<config>/<exp>/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

import openpi.training.config as _config
from openpi.policies import policy_config
from openpi.training.vln_json_dataset import ACTION_TOKEN_MAP, decode_actions

logger = logging.getLogger("pi0_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_STATE_RE = re.compile(
    r"Current state[^:\[]*:\s*\[\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\]"
)

IMAGE_SIZE = 224
STATE_DIM = 4


def _pick_latest_checkpoint(config_name: str) -> str:
    """Return the newest ``<ckpt_base>/<config>/<exp>/<step>`` directory."""
    ckpt_root = Path("checkpoints") / config_name
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoint root: {ckpt_root.resolve()}")

    step_dirs: list[Path] = []
    for exp_dir in ckpt_root.iterdir():
        if not exp_dir.is_dir():
            continue
        for step_dir in exp_dir.iterdir():
            if step_dir.is_dir() and step_dir.name.isdigit() and (step_dir / "params").exists():
                step_dirs.append(step_dir)

    if not step_dirs:
        raise FileNotFoundError(f"No trained step directories with params/ under {ckpt_root.resolve()}")
    step_dirs.sort(key=lambda p: (p.parent.stat().st_mtime, int(p.name)))
    return str(step_dirs[-1].resolve())


def _parse_state(human_text: str) -> np.ndarray:
    m = _STATE_RE.search(human_text)
    if not m:
        return np.zeros((STATE_DIM,), dtype=np.float32)
    return np.asarray(m.groups(), dtype=np.float32)


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (IMAGE_SIZE, IMAGE_SIZE):
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _trim_at_stop(actions: list[str]) -> list[str]:
    for i, a in enumerate(actions):
        if a == "STOP":
            return actions[: i + 1]
    return actions


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
CONFIG_NAME = os.environ.get("OPENPI_CONFIG", "pi0_vln_lora")
CHECKPOINT_DIR = os.environ.get("OPENPI_CHECKPOINT") or _pick_latest_checkpoint(CONFIG_NAME)

logger.info(f"Loading pi0 policy: config={CONFIG_NAME}, checkpoint={CHECKPOINT_DIR}")
_train_config = _config.get_config(CONFIG_NAME)
policy = policy_config.create_trained_policy(_train_config, CHECKPOINT_DIR)
logger.info(
    f"Pi0 policy ready. action_horizon={_train_config.model.action_horizon}, "
    f"action_dim={_train_config.model.action_dim}, "
    f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
)

# pi0's JAX sample_actions is not safe to call concurrently from multiple
# threads sharing one device, so serialize inference inside a backend process.
_infer_lock = threading.Lock()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
class Item(BaseModel):
    prompt: str
    image: str


app = FastAPI()


def _generate(prompt: str, image_path: str) -> list[str]:
    state = _parse_state(prompt)
    image = _load_image(image_path)
    obs = {
        "observation/image": image,
        "observation/state": state,
        "prompt": prompt,
    }
    with _infer_lock:
        out = policy.infer(obs)
    actions_arr = np.asarray(out["actions"])  # (action_horizon, ACTION_DIM)
    action_names = decode_actions(actions_arr)
    return _trim_at_stop(action_names)


@app.post("/generate")
async def generate_endpoint(item: Item) -> list[str]:
    return await run_in_threadpool(_generate, item.prompt, item.image)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "config": CONFIG_NAME,
        "checkpoint": CHECKPOINT_DIR,
        "action_space": list(ACTION_TOKEN_MAP.keys()),
        "action_horizon": _train_config.model.action_horizon,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
