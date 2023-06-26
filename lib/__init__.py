import yaml

with open ("config.yml", "r") as f:
    config = yaml.safe_load(f)
    api_key = config["api_key"]
    dma_api_key = config["dma_api_key"]
    dau_url = config["url"]
    dma_url = config["dma_url"]

