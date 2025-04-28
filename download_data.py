from huggingface_hub import hf_hub_download

for l in ['eng', 'rus', 'tur', 'deu', 'arb']:
  hf_hub_download(repo_id="jumelet/multiblimp", filename=f"{l}/data.tsv", repo_type="dataset", local_dir='hf_cache/')