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

checkpoints = [0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040, 64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140, 64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600, 64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000, 90000, 100000, 110000, 120000, 128000]

api = HfApi()
all_models = api.list_models(author="catherinearnett")
total_downloads = 0
all_bgts = []

for model in tqdm(all_models):
    model_info = api.model_info(model.modelId, expand=["downloadsAllTime"])  
    model_name = str(model.modelId)
    all_bgts.append(model_name)

# create results dataframe
results = pd.DataFrame(columns=['model', 'checkpoint','l1', 'l2', 'biling_cond', 'l1_acc', 'l2_acc'])
results.to_csv('bgpt_multiblimp_results.csv', mode='w', index=False)

# mapping for language codes used for model to language codes used for multiblimp
language_map = {
    'en': 'eng',
    'nl': 'nld',
    'es': 'spa',
    'el': 'ell',
    'pl': 'pol'
}

# loop through all B-GPT models
for m in all_bgts:
    if 'B-GPT' in m:
        # print(m)
        model_cond = m.replace('catherinearnett/B-GPT_', '')
        parts = m.split('_')
        l1 = parts[1]
        l2 = parts[2]
        cond = parts[3]
        l1_iso = language_map[l1]
        l2_iso = language_map[l2]
        
        # loop through checkpoints
        for c in checkpoints:
            m_str = str(m)
            c_str = str(c)
            l1_iso_str = str(l1_iso)
            l2_iso_str = str(l2_iso)
            cond_str = str(cond)

            try:
                # Run the first evaluation for L1
                subprocess.run([
                    "python", "multiblimp/scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--revision", c_str,
                    "--data_dir", f"hf_cache/{l1_iso_str}/",
                    "--src_dir", "multiblimp",
                    "--results_dir", f"bgpt_multiblimp_results/{model_cond}_{c_str}-{l1_iso_str}",
                    "--cache_dir", "hf_cache/"
                ], check=True, env={**os.environ})
                
                # Run the second evaluation for L2
                subprocess.run([
                    "python", "multiblimp/scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--revision", c_str,
                    "--data_dir", f"hf_cache/{l2_iso_str}/",
                    "--src_dir", "multiblimp",
                    "--results_dir", f"bgpt_multiblimp_results/{c_str}-{l2_iso_str}",
                    "--cache_dir", "hf_cache/"
                ], check=True, env={**os.environ})

                # Collect results for L1
                l1_results_path = f"bgpt_multiblimp_results/{model_cond}_{c_str}-{l1_iso_str}/hf_cache_{l1_iso_str}_data.tsv"
                df = pd.read_csv(l1_results_path, sep='\t')
                total_samples = len(df)
                correct_predictions = len(df[df['delta'] > 0])
                l1_accuracy = correct_predictions / total_samples

                # Collect results for L2
                l2_results_path = f"bgpt_multiblimp_results/{c_str}-{l2_iso_str}/hf_cache_{l2_iso_str}_data.tsv"
                df = pd.read_csv(l2_results_path, sep='\t')
                total_samples = len(df)
                correct_predictions = len(df[df['delta'] > 0])
                l2_accuracy = correct_predictions / total_samples

                # Append the new line with results to the CSV file
                new_line = pd.DataFrame({
                    'model': [m_str],
                    'checkpoint': [c],
                    'l1': [l1_iso_str],
                    'l2': [l2_iso_str],
                    'biling_cond': [cond_str],
                    'l1_acc': [l1_accuracy],
                    'l2_acc': [l2_accuracy]
                })
                new_line.to_csv('bgpt_multiblimp_results.csv', mode='a', header=False, index=False)
                print(new_line)
                
            except subprocess.CalledProcessError as e:
                print(f"Error processing model {m_str} at checkpoint {c_str}:")
                print(f"Command failed with exit code {e.returncode}")
                print(f"Error details: {e}")
                
                # Continue with next checkpoint instead of stopping the entire script
                continue
            except Exception as e:
                print(f"Unexpected error with model {m_str} at checkpoint {c_str}:")
                print(f"Error: {e}")
                continue
