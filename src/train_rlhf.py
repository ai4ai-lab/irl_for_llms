# Reference : EleutharAI PPO code scripts
import random
import json
import os
import sys
import sys
from typing import List

import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizer, pipeline

from sklearn.model_selection import train_test_split

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

random.seed(42)

import wandb
wandb.login()

def get_default_config():
    default_config = TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=1000,
            total_steps=1000,
            batch_size=32, #4,
            checkpoint_interval=10000,
            eval_interval=200,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            entity_name='eleutherai',
            project_name='pythia-rlhf',
            save_best=False
        ),
        model=ModelConfig(model_path="EleutherAI/pythia-70m", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-70m", truncation_side="left", padding_side="left",),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1e-6)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=64,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.2,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="running",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=128,
                top_k=30,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )
    return default_config

def get_config(model_size, default_config):
    config_name = model_size
    if config_name == "70M":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/do2vbz2o
        default_config.train.batch_size = 16 
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 1 
        default_config.model.model_path = "lomahony/eleuther-pythia70m-hh-sft"
        default_config.model.num_layers_unfrozen = 4
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-70m/"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-70m"
        default_config.optimizer.kwargs["lr"] = 3e-6 
        default_config.optimizer.kwargs["weight_decay"] = 0.0006
        default_config.scheduler.kwargs["eta_min"] = 5.45e-6
        default_config.method.num_rollouts = 32
        default_config.method.target = 5.71
        default_config.method.ppo_epochs = 8
        default_config.method.chunk_size = 4
        default_config.train.group_name = "EleutherAI/pythia-70m-ppo"
    elif config_name == "160M":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/jubaluv8
        default_config.train.batch_size = 8 
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 10000
        default_config.model.model_path = "lomahony/eleuther-pythia160m-hh-sft"
        default_config.model.num_layers_unfrozen = 4
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/roberta-tox-pythia-160m/"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-160m"
        default_config.optimizer.kwargs["lr"] = 1e-6 # 1.7e-6
        default_config.optimizer.kwargs["weight_decay"] = 3.81e-5
        default_config.scheduler.kwargs["eta_min"] = 1.7e-6
        default_config.method.num_rollouts = 48
        default_config.method.chunk_size = 4
        default_config.method.ppo_epochs = 6
        default_config.method.target = 6.42 
        default_config.train.group_name = "EleutherAI/pythia-160m"
    elif config_name == "410M":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/vpuhppgx
        default_config.train.batch_size = 8
        default_config.train.total_steps = 10000
        default_config.train.seq_length = 10000
        default_config.model.num_layers_unfrozen = 3
        default_config.optimizer.kwargs["lr"] = 8e-7
        default_config.scheduler.kwargs["eta_min"] = 2.2e-7
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-410m"
        default_config.model.model_path = "lomahony/eleuther-pythia410m-hh-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-410m"
        default_config.method.chunk_size = 4
        default_config.method.num_rollouts = 48
        default_config.method.ppo_epochs = 5
        default_config.method.target = 4.9
        default_config.train.group_name = "EleutherAI/pythia-410m"
    elif config_name == "1B":
        default_config.train.batch_size = 4
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 2500
        default_config.optimizer.kwargs["lr"] = 6e-6
        default_config.optimizer.kwargs["weight_decay"] = 0.0002
        default_config.scheduler.kwargs["eta_min"] = 6e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_1B"
        default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
        default_config.method.chunk_size = 16
        default_config.train.group_name = default_config.model.model_path
    elif config_name == "1.4B":
        default_config.train.batch_size = 4
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 1500
        default_config.model.num_layers_unfrozen = 2
        default_config.optimizer.kwargs["lr"] = 6e-6
        default_config.scheduler.kwargs["eta_min"] = 6e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-1.4b"
        default_config.model.model_path = "lomahony/eleuther-pythia1.4b-hh-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-2.8b"
        default_config.method.chunk_size = 4
        default_config.method.num_rollouts = 48
        default_config.method.ppo_epochs = 8
        default_config.method.target = 5.067
        default_config.train.group_name = "EleutherAI/pythia-1.4b-ppo-hptuning"
    elif config_name == "2.8B":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/9ui2aa0x
        default_config.train.batch_size = 2
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 3000
        default_config.model.num_layers_unfrozen = 19
        default_config.optimizer.kwargs["lr"] = 2.3e-6
        default_config.optimizer.kwargs["weight_decay"] = 1.7e-3
        default_config.scheduler.kwargs["eta_min"] = 2.3e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-2.8b"
        default_config.model.model_path = "lomahony/eleuther-pythia2.8b-hh-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-2.8b"
        default_config.method.chunk_size = 4
        default_config.method.num_rollouts = 48
        default_config.method.ppo_epochs = 4
        default_config.method.target = 5
        default_config.train.group_name = "EleutherAI/pythia-2.8b-ppo"
    elif config_name == "6.9B":
        # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/2uue7ywj
        default_config.train.batch_size = 1
        default_config.train.seq_length = 1024
        default_config.train.total_steps = 6000
        default_config.train.checkpoint_dir = "/fsx/orz/pythia-rlhf-checkpoints/ppo_hh/pythia-6.9b"
        default_config.model.model_path = "lomahony/eleuther-pythia6.9b-hh-sft"
        default_config.model.num_layers_unfrozen = 2
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-6.9b"
        default_config.optimizer.kwargs["lr"] = 3.32e-6
        default_config.scheduler.kwargs["eta_min"] = 1e-6
        default_config.method.num_rollouts = 48
        default_config.method.chunk_size = 4
        default_config.method.ppo_epochs = 3
        default_config.method.target = 5.067
        default_config.train.group_name = "EleutherAI/pythia-6.9b-ppo"
    elif config_name == "12B":
        default_config.train.batch_size = 1
        default_config.train.seq_length = 768
        default_config.train.total_steps = 5000
        default_config.optimizer.kwargs["lr"] = 1e-6
        default_config.scheduler.kwargs["eta_min"] = 1e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_pythia_12b"
        default_config.model.model_path = "EleutherAI/pythia-12b"
        default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-12b"
        default_config.method.num_rollouts = 32
        default_config.method.chunk_size = 8
        default_config.train.group_name = default_config.model.model_path
    else:
        raise ValueError(f"Config {config_name} does not exist."
        "Please append this config into the file before calling it")

def get_toxicity_score(scores):
    "Extract value associated with a toxicity score"
    tox_scores = []
    for x in scores:
        if x['label'] == 'toxic':
            tox_scores.append(-x['score'])
        else:
            tox_scores.append(-(1-x['score']))

    return tox_scores

def main(default_config, hparams={}):
    print(f"Config : {default_config}")
    trlx.logging.set_verbosity(trlx.logging.INFO)
    config = TRLConfig.update(default_config, hparams)

    ## Load data
    # Jigsaw
    dataset = load_dataset('jaredjoss/jigsaw-long-2000')["train"]
    all_prompts = [{"prompt": x["prompt"], "original_output": x["original_output"]} for x in dataset]
    prompts, eval_prompts = train_test_split(all_prompts, test_size=0.2, random_state=0)
    
    model_path = 'SkolkovoInstitute/roberta_toxicity_classifier'
    print(model_path)

    rob_tox_fn = pipeline("text-classification", model=model_path, padding="max_length", truncation=True, max_length=512)

    # create reward function
    def reward_fn(samples, prompts, original_output, **kwargs) -> List[float]:
        reward_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        reward_tokenizer.truncation_side = "left"

        samples = [s + reward_tokenizer.eos_token for s in samples]
        rewards = get_toxicity_score(rob_tox_fn(samples))

        original_samples = [p + o + reward_tokenizer.eos_token for p, o in zip(prompts, original_output)]
        original_rewards = get_toxicity_score(rob_tox_fn(original_samples))

        final_rewards = [i-j for i, j in zip(rewards, original_rewards)] 
        return final_rewards

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
    )

    if trainer.accelerator.is_main_process:
        trainer.accelerator.print("\n"*20)
        # trainer.accelerator.print(eval_stats["reward/mean"])

    print("Saving:")
    folder_name = './output/roberta_tox_classifier_custom_jigsaw_70'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        trainer.save_pretrained(folder_name)

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    model_size = '70M'

    wandb.init(project='rlhf_irl', notes=f'Fine tune {model_size}')
    default_config = get_default_config()
    get_config(model_size, default_config)
    main(default_config, hparams)
