from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import os
import torch

# Set the MKL threading layer to GNU to avoid conflicts
os.environ['MKL_THREADING_LAYER'] = 'GNU'

checkpoints = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

api = HfApi()
all_models = api.list_models(author="xiulinyang")
total_downloads = 0
all_bgts = []

for model in tqdm(all_models):
    model_info = api.model_info(model.modelId, expand=["downloadsAllTime"])  
    model_name = str(model.modelId)
    all_bgts.append(model_name)
print(all_bgts)
# create results dataframe
results = pd.DataFrame(columns=['model', 'checkpoint','acc'])
results.to_csv('multiblimp_results.csv', mode='w', index=False)

# mapping for language codes used for model to language codes used for multiblimp
language_map = {
    'EN': 'eng',
    'RU': 'rus',
    'TR': 'tur',
    'DE': 'deu',
    'AR': 'arb'
}

# loop through all B-GPT models
for m in all_bgts:
    if 'GPT-' in m and 'k' in m:
        parts = m.split('-')
        lang = parts[1]
        print(parts)
        vocab_size = parts[2]
        lang_data_id = language_map[lang]
        # loop through checkpoints
        for c in checkpoints:
            m_str = str(m)
            c_str = str(c)
            try:
                # Run the first evaluation for L1
                subprocess.run([
                    "python", "multiblimp/scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--data_dir", f"multiblimp/hf_cache/{lang_data_id}/",
                    "--revision",f'checkpoint-{c_str}',
                    "--src_dir", "multiblimp",
                    "--results_dir", f"multiblimp/multiblimp_results/{lang}_{vocab_size}-{c_str}",
                    "--cache_dir", "multiblimp/hf_cache/"
                ], check=True, env={**os.environ})

                # Collect results for L1
                l1_results_path = f"multiblimp/multiblimp_results/{lang}_{vocab_size}-{c_str}/hf_cache_{lang_data_id}_data.tsv"
                df = pd.read_csv(l1_results_path, sep='\t')
                total_samples = len(df)
                correct_predictions = len(df[df['delta'] > 0])
                l1_accuracy = correct_predictions / total_samples

                # Append the new line with results to the CSV file
                new_line = pd.DataFrame({
                    'model': [m_str],
                    'checkpoint': [c],
                    'acc': [l1_accuracy],

                })
                new_line.to_csv('multiblimp_results.csv', mode='a', header=False, index=False)
                print(new_line)
                
            except subprocess.CalledProcessError as e:
                print(f"Error processing model {m_str} at checkpoint {c_str}:")
                print(f"Command failed with exit code {e.returncode}")
                print(f"Error details: {e}")
                
                # Continue with next checkpoint instead of stopping the entire script
                continue
