# 🗺️ AirNav: A Large-Scale Real-World UAV Vision-and-Language Navigation Dataset with Natural and Diverse Instructions

---

## 📑 Introduction

Existing UAV VLN datasets face issues such as dependence on virtual environments, lack of naturalness in instructions, and limited scale.
To address these challenges, we propose AirNav, a large-scale UAV VLN benchmark based on real urban aerial images, which provides natural and diverse instructions. Additionally, we introduce the AirVLN-R1, which combines Supervised Fine-Tuning and Reinforcement Fine-Tuning strategies to significantly enhance performance and generalization ability. The feasibility of the model has been validated through real-world tests. Our dataset and code are open-source and available for community use.

## 🛠️ Environment Setup

This project depends on multiple models and tool libraries. It is recommended to use Conda to create an isolated environment.

### Install Conda Environment

```bash
conda create -n airnav python=3.10
conda activate airnav

pip install -r requirements.txt
```

---

## 🏗️ Data Generation Pipeline

The `data_generation/` module generates natural and diverse AirNav-style navigation instructions through a four-stage pipeline.

1. Start/target sampling and target description generation
2. Landmark planning and description refinement
3. Trajectory synthesis with action sequences
4. Multi-persona instruction generation

Run from project root:

```bash
python data_generation/data_pipeline.py
```

Before running, create `data_generation/config.yaml` and set:

- dataset paths (`citynav_data_path`, `citynav_data_info_path`)
- output paths (`landmark_save_path`, `landmark_revised_path`, `instruction_save_path`)
- API config (`api.gpt4o`)

Main outputs:

- `landmark_data.json` (step 1-3 results)
- `landmark_data_revised.json` (refined landmarks)
- `instruction_persona.json` (final instructions)

## 📦 Model and Data Preparation

### Dataset Structure

🔗 **Download link:** [AirNav](https://huggingface.co/datasets/dpairnav/AirNav)

- Download data to `./data/`
- The `AirNav` dataset is organized into `train`, `val`, and `test` splits as follows:

```text
data
|-- AirNav
|   |-- test
|   |   |-- airnav_test.json
|   |   |-- info_test.json
|   |-- train
|   |   |-- airnav_train.json
|   |   `-- info_train.json
|   `-- val
|       |-- airnav_val_seen.json
|       |-- airnav_val_unseen.json
|       |-- info_val_seen.json
|       `-- info_val_unseen.json
|-- cityrefer
|   ...
|-- gsam
|   ...
`-- rgbd-new
|   ...
```

**File Description**

- `airnav_*.json` files specify the environment configuration and are used to initialize the navigation simulator.
- `info_*.json` files provide navigation instructions, action annotations, and associated landmark information for each episode.

### Model Weights

- Download model weights to `./model_weight/`

  | Baselines         | NE(m) | SR(%) | OSR(%) | SPL(%) | Checkpoints                                         |
  | ----------------- | ----- | ----- | ------ | ------ | --------------------------------------------------- |
  | Seq2Seq           | 336.1 | 1.28  | 10.31  | 1.08   | [💾](https://huggingface.co/dpairnav/AirNavSeq2Seq) |
  | CMA               | 190.3 | 4.48  | 17.06  | 4.03   | [💾](https://huggingface.co/dpairnav/AirNavCMA)     |
  | Qwen2.5-VL-7B SFT | 48.3  | 39.56 | 52.41  | 38.53  | [💾](https://huggingface.co/dpairnav/AirNavSFT)     |
  | Qwen2.5-VL-7B RL  | 165.8 | 2.31  | 4.39   | 2.03   | [💾](https://huggingface.co/dpairnav/AirNavRL)      |
  | AirVLN-R1         | 40.0  | 51.75 | 62.29  | 50.57  | [💾](https://huggingface.co/dpairnav/AirVLN-R1)     |


## 🧠 Inference

1. Option A: Start the local vLLM service

**Note:** This project has been tested with **vLLM v0.7.3**, and using this version is recommended for best compatibility.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve path/to/your/model \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen_2_5_vl_7b \
  --host 0.0.0.0 \
  -tp 4 \
  --uvicorn-log-level debug \
  --port 8000 \
  --limit-mm-per-prompt image=5,video=0 \
  --max-model-len=4096
```

2. Option B: Use a GPT-style API service (no local vLLM)

Configure the API client and model in `eval.py`:

- `GPT_client`: set your API key and endpoint
- `GPT_model`: set a vision-capable model name
- `MODEL_TYPE = GPT_model`

Model/API settings used in `eval.py`:

| API Provider | Model | Version | Temperature |
| --- | --- | --- | --- |
| Azure OpenAI | `gpt-5` | `2024-12-01-preview` | `1.0` |
| Azure OpenAI | `gpt-4o` | `2024-12-01-preview` | `1.0` |
| DashScope (compatible-mode) | `qwen3-vl-plus` | `empty` | `1.0` |

3. Start the inference script

```bash
python eval.py
```

4. Result Visualization
  All intermediate visualization images, as well as the final UAV flight trajectory visualization, will be saved in the `EvalPhotoData` directory.

---

## 🚀 Training

⚠️ **Prerequisites**: Please configure the environments for **LLaMA-Factory** and **VERL** before training.
Each framework needs its own Python environment (different dependency pins).

For LLaMA-Factory, we recommend a Python 3.10 conda env with the versions in
`LLaMA-Factory/requirements.txt`:
```bash
conda create -p /path/to/env/air_train python=3.10 -y && conda activate /path/to/env/air_train
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd LLaMA-Factory && pip install -r requirements.txt && pip install -e . --no-deps
```

For VERL, we recommend the official install script (FSDP-only mode), then add the
reward-function dependency:
```bash
conda create -p /path/to/env/air_rl python=3.10 -y && conda activate /path/to/env/air_rl
cd verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install rasterio   # required by verl/reward_fn/AirNav_rl.py
```

1. **Training Data Preparation**
  The `train_data_generate.py` script transforms the raw data into training-ready data.
  All training-related images are stored in the `TrainPhotoData` directory.
  The resulting training data should be further processed into formats compatible with the **LLaMA-Factory** and **VERL** frameworks for subsequent training.

```bash
python train_data_generate.py
```

For LLaMA-Factory, convert the output of `train_data_generate.py` to ShareGPT schema:

```bash
python train_to_sharegpt.py --input data/AirNav/train/train.json --output LLaMA-Factory/data/airnav_sft.json
```

Then register the output of `train_to_sharegpt.py` in `LLaMA-Factory/data/dataset_info.json`.

For VERL, convert the output of `train_data_generate.py` into VERL parquet format. 

```bash
cd verl
python examples/data_preprocess/AirNav_tool.py --local_dir ./data
```

2. **SFT**

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/AirNav_lora_sft.yaml
```


3. **GRPO**

```bash
bash ./my_script/run_qwen2_5_vl_7b.sh
```

