import yaml

with open('/root/nlp_project/LLaMA-Factory/examples/inference/llama3_vllm.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)
