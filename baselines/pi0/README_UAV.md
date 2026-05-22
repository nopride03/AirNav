# Training and Serving pi0 for UAV Navigation

This guide explains how to fine-tune this repository on a UAV vision-language navigation task and how to serve the trained policy as an HTTP API.

The UAV task uses the `pi0_vln_lora` and `pi0_vln_smoke` training configs. They read an OpenVLA-style JSON dataset through the custom `vln_json:` data loader.

## 1. Environment Setup

Install the project dependencies with `uv`:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

By default, the helper scripts look for a virtual environment at `.venv`. You can override it:

```bash
export OPENPI_ENV=/path/to/your/python/env
```

The training config loads the pi0 base checkpoint from the public Google Cloud path by default:

```bash
export OPENPI_PI0_BASE_CHECKPOINT=gs://openpi-assets/checkpoints/pi0_base/params
```

If you keep a local copy of the checkpoint, set `OPENPI_PI0_BASE_CHECKPOINT` to the local `params` directory.

## 2. Prepare the UAV Dataset

Put the converted UAV navigation JSON files under a local data directory, for example:

```text
data/vln/
  train_converted.json
  train_1024_converted.json
```

Then point the training scripts to that directory:

```bash
export OPENPI_VLN_DATA_DIR=data/vln
```

Each JSON item should contain an image path and a conversation pair:

```json
{
  "image": "/absolute/path/to/cur_view.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n... Current state of the UAV: [x, y, z, heading] ..."
    },
    {
      "from": "gpt",
      "value": "['MOVE_FORWARD', 'TURN_LEFT', 'STOP']"
    }
  ]
}
```

The supported discrete actions are:

```text
STOP
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
```

The loader parses the UAV state from the human prompt, loads the image, and converts the action list into the continuous action representation used by pi0.

## 3. Run a Smoke Test

Before full training, run the small smoke config on `train_1024_converted.json`:

```bash
bash train_pi0.sh smoke
```

Useful overrides:

```bash
BATCH_SIZE=4 NUM_EPOCHS=1 bash train_pi0.sh smoke
```

This validates that the dataset, image paths, checkpoint loading, and training loop work end to end.

## 4. Train the UAV LoRA Policy

Run LoRA fine-tuning on the full UAV dataset:

```bash
bash train_pi0.sh lora
```

Common options:

```bash
NUM_EPOCHS=2 BATCH_SIZE=8 EXP_NAME=uav-lora bash train_pi0.sh lora
```

For multi-GPU training, set `FSDP_DEVICES` to the number of devices used for model sharding:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FSDP_DEVICES=4 bash train_pi0.sh lora
```

The script computes the number of optimizer steps from:

```text
number of samples / batch size * number of epochs
```

Checkpoints are saved under:

```text
checkpoints/pi0_vln_lora/<experiment_name>/<step>/
```

Each completed step directory should contain a `params/` subdirectory.

## 5. Serve the Trained Policy as an API

After training, start the multi-GPU inference launcher:

```bash
GPU_LIST=0,1,2,3 bash start_all.sh
```

By default, this starts:

```text
one backend per GPU on ports 8050, 8051, ...
a round-robin gateway on port 9000
```

If `OPENPI_CHECKPOINT` is not set, `api.py` automatically picks the latest checkpoint under:

```text
checkpoints/pi0_vln_lora/
```

To serve a specific checkpoint:

```bash
export OPENPI_CONFIG=pi0_vln_lora
export OPENPI_CHECKPOINT=checkpoints/pi0_vln_lora/uav-lora/12345
GPU_LIST=0 bash start_all.sh
```

You can also change ports and host binding:

```bash
HOST=0.0.0.0 GATEWAY_PORT=9000 BASE_PORT=8050 GPU_LIST=0,1 bash start_all.sh
```

## 6. Call the API

Send requests to the gateway:

```bash
curl -X POST http://127.0.0.1:9000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "<image>\n... Current state of the UAV: [330.3, 461.3, 60.0, -149.8] ...",
    "image": "/absolute/path/to/cur_view.jpg"
  }'
```

The response is a JSON list of UAV actions:

```json
["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]
```

The server predicts up to 8 future actions and truncates the output at the first `STOP`.

Health checks:

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/backends
```

Backend logs are written to:

```text
logs/backend_gpu<gpu>_port<port>.log
logs/gateway_port<port>.log
```

## 7. AirNav Benchmark Evaluation

```bash
# Run evaluation with default settings
PI0_URL=http://127.0.0.1:9000 python eval_pi0.py
```
