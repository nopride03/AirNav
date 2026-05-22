"""Input / output transforms for VLN datasets consumed by pi0.

The transforms convert the sample dict produced by
``openpi.training.vln_json_dataset.VLNJsonDataset`` (keys: ``image``, ``state``,
``actions``, ``prompt``) into the structure expected by the pi0 model, and on
the inference side slice the model's padded action vector back down to the
single scalar we care about (mapped to STOP / MOVE_FORWARD / TURN_LEFT /
TURN_RIGHT via ``ACTION_TOKEN_MAP``).

The transforms mirror the structure of ``libero_policy.py`` so that the rest of
the openpi training pipeline (normalisation, prompt tokenisation, padding) can
be reused unmodified.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.training import vln_json_dataset as _vln_ds


def make_vln_example() -> dict:
    """Creates a random input example for the VLN policy (useful for inference tests)."""
    return {
        "observation/state": np.zeros(4, dtype=np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "move forward to the target and stop",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class VLNInputs(transforms.DataTransformFn):
    """Input transform for VLN (UAV top-down view, 4-d state, 1-d discretised actions).

    The dataset only has a single camera view, so ``base_0_rgb`` is the actual
    image and the two wrist-camera slots are zero-padded (matching how Libero
    pads the right wrist).
    """

    model_type: _model.ModelType

    RAW_ACTION_DIM: ClassVar[int] = _vln_ds.ACTION_DIM

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs: dict = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # We only mask padding images for pi0, not pi0-FAST (matching libero_policy.py).
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class VLNOutputs(transforms.DataTransformFn):
    """Output transform: slice the padded action vector down to the single VLN dim."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., : VLNInputs.RAW_ACTION_DIM]}
