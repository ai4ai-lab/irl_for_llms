# Inverse Reinforcement Learning for Large Language Models

This repository implements **Inverse Reinforcement Learning (IRL)** for extracting reward models from **Reinforcement Learning from Human Feedback (RLHF)**-fine-tuned Large Language Models (LLMs). The project includes scripts for fine-tuning LLMs, creating IRL datasets, and applying Max-Margin IRL. If you find this code helpful, please cite our paper "Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse Reinforcement Learning" using the following:

## 📚 Citation
```bibtex
@article{joselowitz2024insights,
  title={Insights from the inverse: Reconstructing LLM Training Goals through Inverse RL},
  author={Joselowitz, Jared and Majumdar, Ritam and Jagota, Arjun and Bou, Matthieu and Patel, Nyal and Krishna, Satyapriya and Parbhoo, Sonali},
  journal={In Proceedings of 2nd Conference on Language Modelling},
  pages={arXiv--2410},
  year={2025},
  url={https://arxiv.org/abs/2410.12491}
}
```

- This is the joint work of Jared Joselowitz, Ritam Majumdar, Arjun Jagota, Matthieu Bou, Nyal Patel, Satyapriya Krishna, Sonali Parbhoo
- Accepted at The Conference on Language Modeling (COLM) 2025

## Abstract
Large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF) have demonstrated remarkable capabilities, but their underlying reward functions and decision-making processes remain opaque. This paper introduces a novel approach to interpreting LLMs by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. We conduct experiments on toxicity-aligned LLMs of varying sizes, extracting reward models that achieve up to 85\% accuracy in predicting human preferences. Our analysis reveals key insights into the non-identifiability of reward functions, the relationship between model size and interpretability, and potential pitfalls in the RLHF process. We demonstrate that IRL-derived reward models can be used to fine-tune new LLMs, resulting in comparable or improved performance on toxicity benchmarks. This work provides a new lens for understanding and improving LLM alignment, with implications for the responsible development and deployment of these powerful systems.


## Environment Setup

To replicate the environment used for training, follow the steps below:

### Step 1: Create and activate a virtual environment
```bash
conda create --name IRLforLLM python=3.11.7
conda activate IRLforLLM
```

### Step 2: Clone this repository
```bash
git clone git@github.com:ai4ai-lab/irl_for_llms.git
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
