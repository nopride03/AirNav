"""VLN JSON dataset for pi0 training.

Bridges the OpenVLA-style converted JSON (same file consumed by the OpenFly /
OpenVLA / NaVid baselines, for example ``data/vln/train_converted.json``)
into the openpi training pipeline, without requiring a full LeRobot conversion.

Each raw JSON sample looks like::

    {
        "image": "/abs/path/to/cur_view.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\\n... Navigation instruction: ... [x, y, z, heading] ..."},
            {"from": "gpt",   "value": "['MOVE_FORWARD', 'MOVE_FORWARD', 'STOP']"}
        ]
    }

We convert each sample on-the-fly into the dict expected by openpi (after the
``repack_transforms`` and ``data_transforms`` stages):

    {
        "image":   uint8 ndarray of shape (H, W, 3),
        "state":   float32 ndarray of shape (4,)  -- [x, y, z, heading],
        "actions": float32 ndarray of shape (action_horizon, 3)
                   -- 3-channel continuous action code, see ``ACTION_TOKEN_MAP``.
                      Each channel is in [-1, 1] and the four VLN actions are
                      spread across 4 corners of the 3-d cube so pi0's L2
                      flow-matching loss can naturally separate them.
        "prompt":  str  -- the full human-turn text from the conversation
                   (task template + instruction + state description + action
                   space rules). This matches what the OpenFly baseline feeds
                   into its LLM.
    }
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ACTION_TOKEN_MAP: dict[str, np.ndarray] = {
    "MOVE_FORWARD": np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
    "TURN_LEFT": np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
    "TURN_RIGHT": np.asarray([-1.0, 1.0, -1.0], dtype=np.float32),
    "STOP": np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
}

ACTION_DIM: int = 3
STATE_DIM: int = 4

_ACTION_NAMES: list[str] = list(ACTION_TOKEN_MAP.keys())
_ACTION_CODES: np.ndarray = np.stack(list(ACTION_TOKEN_MAP.values()), axis=0)  # (num_actions, 3)

_STATE_RE = re.compile(
    r"Current state[^:\[]*:\s*\[\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\]"
)


def _extract_human_turn(example: dict[str, Any]) -> str:
    for turn in example["conversations"]:
        if turn["from"] == "human":
            return turn["value"].replace("<image>\n", "", 1).replace("<image>", "", 1).strip()
    raise ValueError("Missing human turn in example")


def _extract_action_sequence(example: dict[str, Any]) -> list[str]:
    for turn in example["conversations"]:
        if turn["from"] != "gpt":
            continue
        value = turn["value"]
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, list):
            raise ValueError(f"Expected action list, got `{type(value)}`")
        return value
    raise ValueError("Missing gpt turn in example")


def _parse_state(human_text: str) -> np.ndarray:
    m = _STATE_RE.search(human_text)
    if not m:
        return np.zeros((STATE_DIM,), dtype=np.float32)
    return np.asarray(m.groups(), dtype=np.float32)


def encode_actions(action_sequence: list[str], action_horizon: int) -> np.ndarray:
    """Map up to ``action_horizon`` discrete actions to an
    ``(action_horizon, ACTION_DIM)`` float32 array. Pads with ``STOP``
    if the input sequence is shorter than ``action_horizon``.
    """
    stop = ACTION_TOKEN_MAP["STOP"]
    out = np.broadcast_to(stop, (action_horizon, ACTION_DIM)).copy()
    for i, name in enumerate(action_sequence[:action_horizon]):
        if name not in ACTION_TOKEN_MAP:
            raise ValueError(f"Unsupported action `{name}`")
        out[i] = ACTION_TOKEN_MAP[name]
    return out


def decode_actions(action_chunk: np.ndarray) -> list[str]:
    """Nearest-neighbour decoder: turn a continuous ``(action_horizon, >=ACTION_DIM)``
    array back into discrete action names (STOP / MOVE_FORWARD / TURN_LEFT /
    TURN_RIGHT) using L2 distance to the corners defined in ``ACTION_TOKEN_MAP``.
    Extra trailing channels (from pi0's padded action_dim=32) are ignored.
    """
    arr = np.asarray(action_chunk, dtype=np.float32)[..., :ACTION_DIM]
    # squared L2 distance to each code -> (..., num_actions)
    diff = arr[..., None, :] - _ACTION_CODES[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    idx = np.argmin(dist_sq, axis=-1)
    return [_ACTION_NAMES[i] for i in np.atleast_1d(idx).tolist()]


class VLNJsonDataset(Dataset):
    """Torch Dataset reading ``train_converted.json`` for pi0 training.

    Images are decoded from disk lazily; keeps the in-memory footprint to just
    the parsed JSON (paths + text).
    """

    def __init__(
        self,
        data_json_path: str | Path,
        action_horizon: int,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_json_path = Path(data_json_path)
        if not self.data_json_path.exists():
            raise FileNotFoundError(f"VLN JSON not found: {self.data_json_path}")
        logger.info(f"Loading VLN json from {self.data_json_path}")
        with self.data_json_path.open("r") as f:
            self.examples: list[dict[str, Any]] = json.load(f)
        logger.info(f"VLN json examples: {len(self.examples)}")
        self.action_horizon = action_horizon
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.examples)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.data_json_path.parent / path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]

        img_path = self._resolve_image_path(ex["image"])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            if im.size != (self.image_size, self.image_size):
                im = im.resize((self.image_size, self.image_size), Image.BILINEAR)
            image = np.asarray(im, dtype=np.uint8)

        human = _extract_human_turn(ex)
        state = _parse_state(human)
        actions = encode_actions(_extract_action_sequence(ex), self.action_horizon)

        return {
            "image": image,
            "state": state,
            "actions": actions,
            "prompt": human,
        }
