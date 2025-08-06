# Inverse Reinforcement Learning for Large Language Models

This repository implements **Inverse Reinforcement Learning (IRL)** for extracting reward models from **Reinforcement Learning from Human Feedback (RLHF)**-fine-tuned Large Language Models (LLMs). The project includes scripts for fine-tuning LLMs, creating IRL datasets, and applying Max-Margin IRL. If you find this code helpful, please cite our paper "Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse Reinforcement Learning" using the following:

## 📚 Citation
```bibtex
@article{joselowitz2024insights,
  title={Insights from the inverse: Reconstructing LLM Training Goals through Inverse RL},
  author={Joselowitz, Jared and Jagota, Arjun and Bou, Matthieu and Patel, Nyal and Krishna, Satyapriya and Parbhoo, Sonali},
  journal={In Proceedings of 2nd Conference on Language Modelling},
  pages={arXiv--2410},
  year={2025},
  url={https://arxiv.org/abs/2410.12491}
}
```


## Environment Setup

To replicate the environment used for training, follow the steps below:

### Step 1: Create and activate a conda environment
```bash
conda create --name IRLforLLM python=3.11.7
conda activate IRLforLLM
```

### Step 2: Clone this repository
```bash
git clone git@github.com:JaredJoss/irl_for_llms.git
cd irl_for_llms
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

To fine-tune LLMs using RLHF, you'll need the trlx library. Follow the instructions on the official TRLx GitHub page to install it:

[https://github.com/CarperAI/trlx](https://github.com/CarperAI/trlx)


## Usage

### Fine-Tuning an LLM using RLHF
To fine-tune a large language model using RLHF, use the following command:
```bash
python src/train_rlhf.py
```

### Generate IRL Demonstrations
Before running the IRL algorithm, you need to generate demonstrations using the original and RLHF-trained model. Run the following script to create the necessary dataset for IRL:

```bash
python src/create_dataset_irl.py
```

### Extract Reward Model using Max-Margin IRL
After generating the demonstrations, you can implement Max-Margin IRL and extract the reward function from the RLHF-trained LLM:

```bash
python src/irl.py
```

### Run RLHF using IRL extracted reward model
After extracting the reward model from the RLHF'd LLM, you can use this reward model to fine-tune other LLMs:
```bash
python src/train_rlhf_irl_rm.py
```
