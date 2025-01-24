from datasets import load_dataset
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluate
import os

toxicity = evaluate.load("toxicity")

def create_model_continuations(toxic_prompts, text_generation_pipeline):
    model_continuations=[]
    for prompt in toxic_prompts:
        generation = text_generation_pipeline(prompt, max_new_tokens=50, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(prompt, '').strip()
        model_continuations.append(continuation)

    print('Generated ' + str(len(model_continuations)) + ' continuations')
    
    return model_continuations

def load_and_evaluate(model_name, prompts):
    # load model
    text_generation_pipe = pipeline("text-generation", model=model_name)
    # evaluate toxicity
    model_continuations = create_model_continuations(prompts, text_generation_pipe)
    # remove empty strings
    model_continuations = [x for x in model_continuations if len(x) > 2]
    toxicity_ratio = toxicity.compute(predictions=model_continuations, aggregation="ratio")

    return toxicity_ratio, model_continuations

def calc_mean_toxicity(model_continuations):
    tox_dict = {}
    all_toxicity = toxicity.compute(predictions=model_continuations)
    for text, score in zip(model_continuations, all_toxicity['toxicity']):
        tox_dict[text] = score
    tox_dict = (dict(sorted(tox_dict.items(), key=lambda item: item[1], reverse=True)))
    toxicity_scores = [tox for tox in tox_dict.values()]
    mean_toxicity = np.mean(toxicity_scores)

    return mean_toxicity, tox_dict

## Load dataset
dataset = load_dataset('jaredjoss/jigsaw-long-2000')["train"]
all_prompts = [{"prompt": x["prompt"], "original_output": x["original_output"]} for x in dataset]
prompts, eval_prompts = train_test_split(all_prompts, test_size=0.2, random_state=0)
prompts_test = [x['prompt'] for x in eval_prompts]

## Load models 
fol_name = 'RLHF/rlhf_70m'
models = [f'{fol_name}/{file}' for file in os.listdir(fol_name) if not file.startswith('.')]           
models[:0] = []

## Evaluate
# Check if the CSV file exists
csv_file_path = f'output/toxicity_results/{fol_name.split("/")[1]}_toxicity_results.csv'
if os.path.exists(csv_file_path):
    # If the CSV file exists, load it and filter out already analyzed models
    existing_df = pd.read_csv(csv_file_path)
    existing_models = existing_df['Model'].tolist()
    models_to_analyze = [model for model in models if model.split('/')[-1] not in existing_models]
    df = existing_df
else:
    # If the CSV file doesn't exist, analyze all models
    models_to_analyze = models
    df = pd.DataFrame(columns=['Model', 'Toxicity Ratio', 'Mean Toxicity', 'Model Continuations'])

tox_ratios, mean_toxicities = {}, {}
for model in models_to_analyze:
    print("\nModel:  ", model)

    toxicity_ratio, model_continuations = load_and_evaluate(model, prompts_test)
    mean_toxicity, tox_dict = calc_mean_toxicity(model_continuations)
    tox_ratios[model] = toxicity_ratio
    mean_toxicities[model] = mean_toxicity
    data = {
        'Model': [model.split('/')[-1]],
        'Toxicity Ratio': [toxicity_ratio],
        'Mean Toxicity': [mean_toxicity],
        'Model Continuations': [model_continuations]
    }
    new_row = pd.DataFrame(data)
    df = pd.concat([df, new_row], ignore_index=True)

    print(f'Toxic ratio: {toxicity_ratio}')
    print(f'Mean toxicity: {mean_toxicity}')

df.to_csv(csv_file_path, index=False)