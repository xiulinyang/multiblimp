from huggingface_hub import hf_hub_download

for l in ['eng', 'tur', 'arb', 'deu', 'rus']:  # Loop over language codes
    hf_hub_download(
        repo_id="jumelet/multiblimp",          # Repository on Huggingface
        filename=f"{l}/data.tsv",               # Which file to download
        repo_type="dataset",                    # It's a dataset, not a model
        local_dir='hf_cache/'                   # Save here locally
    )
