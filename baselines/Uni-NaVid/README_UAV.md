# Training and Serving UAV Navigation Tasks

This guide explains how to fine-tune Uni-NaVid for UAV navigation tasks and deploy the trained checkpoint as an HTTP API.

## 1. Environment Setup

Create and activate the Python environment, then install the package:

```bash
conda create -n uninavid python=3.10 -y
conda activate uninavid
pip install --upgrade pip
pip install -e .
pip install flash-attn==2.5.9.post1
```

Download the required model files into `model_zoo/`:

```text
model_zoo/
├── eva_vit_g.pth
└── uninavid-7b-full-224-video-fps-1-grid-2/
```

The `checkpoints/`, `data/`, and `model_zoo/` directories are ignored by git because they usually contain large private files.

## 2. Prepare UAV Training Data

The training script expects Uni-NaVid/LLaVA-style conversation JSON. Each sample should contain an image path and a conversation:

```json
{
  "image": "path/to/current_view.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n## Role\nYou are an expert navigation assistant for a UAV flight simulator.\n\n## Text Input\n- Navigation instruction: Move forward and stop near the target.\n- Current state of the UAV: [x, y, z, heading]\n\nPredict the next UAV actions."
    },
    {
      "from": "gpt",
      "value": "['MOVE_FORWARD', 'MOVE_FORWARD', 'STOP']"
    }
  ]
}
```

Place the training JSON under `data/`, for example:

```text
data/
└── train_converted.json
```

The image paths inside the JSON can be absolute paths or paths that are valid on the training machine. For public releases, keep private images and raw datasets outside git.

## 3. Fine-Tune on UAV Data

Fine-tune from the released Uni-NaVid checkpoint:

```bash
PREV_MODEL=./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
DATA_PATH=./data/train_converted.json \
MODEL_PATH=./checkpoints/uninavid \
bash scripts/uninavid_stage_2.sh
```

If you prefer to use the wrapper script:

```bash
CONDA_ENV=uninavid bash train.sh
```

Important variables:

- `PREV_MODEL`: base Uni-NaVid checkpoint used for fine-tuning.
- `DATA_PATH`: UAV training JSON.
- `MODEL_PATH`: output directory for the trained checkpoint.

The default stage-2 training script uses DeepSpeed `scripts/zero2.json`, BF16, batch size 8 per device, and saves checkpoints every 2000 steps. Adjust `per_device_train_batch_size`, `gradient_accumulation_steps`, and `save_steps` in `scripts/uninavid_stage_2.sh` for your hardware.

## 4. Deploy the Trained Checkpoint as an API

After training, start a single API worker:

```bash
python api.py \
  --host 0.0.0.0 \
  --port 8050 \
  --model-path ./checkpoints/uninavid
```

Generate actions:

```bash
curl -X POST http://127.0.0.1:8050/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<image>\nYour UAV navigation instruction here.",
    "image": "path/to/current_view.jpg"
  }'
```

The response is a parsed action list, for example:

```json
["MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "STOP"]
```

## 5. Multi-GPU API Gateway

Use `api.sh` to launch one backend worker per GPU and a gateway that load-balances requests:

```bash
GPU_LIST=0,1,2,3 \
MODEL_PATH=./checkpoints/uninavid \
BASE_PORT=8050 \
GATEWAY_PORT=9000 \
bash api.sh
```

Useful variables:

- `GPU_LIST`: comma-separated GPU ids, for example `0,1,2,3`.
- `MODEL_PATH`: trained checkpoint path.
- `BASE_PORT`: first backend port. Each GPU uses `BASE_PORT + index`.
- `GATEWAY_PORT`: public gateway port.
- `HOST`: bind address, defaults to `0.0.0.0`.
- `TIMEOUT`: gateway request timeout in seconds.

Send requests to the gateway:

```bash
curl -X POST http://127.0.0.1:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<image>\nYour UAV navigation instruction here.",
    "image": "path/to/current_view.jpg"
  }'
```

## 6. Quick Python Client

```python
import requests

response = requests.post(
    "http://127.0.0.1:9000/generate",
    json={
        "prompt": "<image>\nYour UAV navigation instruction here.",
        "image": "path/to/current_view.jpg",
    },
    timeout=120,
)
response.raise_for_status()
print(response.json())
```

## 7. Notes

- Keep private datasets, trained checkpoints, and raw images outside git or under ignored directories such as `data/`, `checkpoints/`, and `model_zoo/`.
- The API reads image files from the server filesystem. The `image` field must be a path accessible to the machine running the API.
- The constrained action decoder in `api.py` currently uses `STOP`, `MOVE_FORWARD`, `TURN_RIGHT`, and `TURN_LEFT`. Update `ACTION_SPACE` if your UAV task uses a different action set.

## 8. AirNav Benchmark Evaluation

```bash
# Run evaluation with default settings
UNINAVID_URL=http://127.0.0.1:9000 python eval_uninavid.py
```


