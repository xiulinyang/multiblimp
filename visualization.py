import pandas as pd
import matplotlib.pyplot as plt
import re

# Load CSV without header, assign column names
multiblimp_results = pd.read_csv('multiblimp_results_3000.csv', header=None, names=['model', 'checkpoint', 'accuracy'])

# Remove the "xiulinyang/" prefix
multiblimp_results['model'] = multiblimp_results['model'].str.replace('xiulinyang/', '', regex=False)

# Extract language code (e.g., EN, RU, TR, DE) from model name
multiblimp_results['language'] = multiblimp_results['model'].str.extract(r'-([A-Z]{2})-')

# Group by language
languages = multiblimp_results['language'].unique()
n_langs = len(languages)

# Create subplots: one per language
fig, axes = plt.subplots(1,n_langs,  figsize=(8 * n_langs,8), sharex=True, sharey=True)


if n_langs == 1:
    axes = [axes]  # ensure axes is always iterable

for ax, lang in zip(axes, languages):
    lang_data = multiblimp_results[multiblimp_results['language'] == lang]

    for model_name, group in lang_data.groupby("model"):
        print(model_name)
        ax.plot(group["checkpoint"], group["accuracy"], marker='o', label=model_name)
    ax.set_title(f"{lang}")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend(title="Model")

axes[-1].set_xlabel("Checkpoint")
plt.tight_layout()
plt.savefig('multiblimp_results1.pdf')
plt.show()
