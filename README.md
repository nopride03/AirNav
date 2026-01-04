# 🗺️ AirNav: A Large-Scale Real-World UAV Vision-and-Language Navigation Dataset with Natural and Diverse Instructions

---

## 📑 Introduction

Existing UAV VLN datasets face issues such as dependence on virtual environments, lack of naturalness in instructions, and limited scale.
To address these challenges, we propose AirNav, a large-scale UAV VLN benchmark based on real urban aerial images, which provides natural and diverse instructions. Additionally, we introduce the AirVLN-R1, which combines Supervised Fine-Tuning and Reinforcement Fine-Tuning strategies to significantly enhance performance and generalization ability. The feasibility of the model has been validated through real-world tests. Our dataset and code are open-source and available for community use.

## 🛠️ Environment Setup

This project depends on multiple models and tool libraries. It is recommended to use Conda to create an isolated environment.

### Install Conda Environment

```bash
- conda create -n airnav python=3.10
- conda activate airnav

- pip install -r requirements.txt
```

---

## 📦 Model and Data Preparation

### Dataset Structure

The AirNav dataset is currently under preparation and will be released upon paper acceptance.

🔗 **Download link:** *Coming soon*

* Download data to `./data/`
* The `AirNav` dataset is organized into `train`, `val`, and `test` splits as follows:

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

- **`airnav_*.json`** files specify the environment configuration and are used to initialize the navigation simulator.
- **`info_*.json`** files provide navigation instructions, action annotations, and associated landmark information for each episode.

### Model Weights

* Download model weights to `./model_weight/`
  
  | Baselines         | NE(m) | SR(%) | OSR(%) | SPL(%) | Checkpoints                                                                                                    |
  | ----------------- | ----- | ----- | ------ | ------ | -------------------------------------------------------------------------------------------------------------- |
  | Seq2Seq           | 336.1 | 1.28  | 10.31   | 1.08   | 💾 *TBA*   |
  | CMA               | 190.3 | 4.48  | 17.06   | 4.03   | 💾 *TBA* |
  | Qwen2.5-VL-7B SFT | 48.3 | 39.56  | 52.41  | 38.53   | 💾 *TBA*      |
  | Qwen2.5-VL-7B RL  | 165.8 | 2.31  | 4.39   | 2.03   | 💾 *TBA*    |
  | AirVLN-R1         | 40.0 | 51.75  | 62.29  | 50.57   | 💾 *TBA*       |

## 🧠 Inference

1. Start the vLLM service

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve path/to/your/model \
  --dtype auto \
  --trust-remote-code \
  --served-model-name qwen_2_5_vl_7b \
  --host 0.0.0.0 \
  -tp 4 \
  --uvicorn-log-level debug \
  --port your_port \
  --limit-mm-per-prompt image=5,video=0 \
  --max-model-len=4096
```

2. Start the inference script

```bash
python eval.py
```

3. Result Visualization
   All intermediate visualization images, as well as the final UAV flight trajectory visualization, will be saved in the `EvalPhotoData` directory.

---

## 🚀 Training

⚠️ **Prerequisites**: Please configure the environments for LLaMA-Factory and VERL before training.

1. **Training Data Preparation**

  The `train_data_generate.py` script transforms the raw data into training-ready data.
  All training-related images are stored in the `TrainPhotoData` directory.
  The resulting training data should be further processed into formats compatible with the **LLaMA-Factory** and **VERL** frameworks for subsequent training.

```bash
python train_data_generate.py
```

2. **SFT**

```bash
cd LLaMA-Factory
llamafactory-cli train examples/train_lora/AirNav_lora_sft.yaml
```

3. **GRPO**

```bash
cd verl
bash ./my_script/run_qwen2_5_vl_7b.sh
```

---
