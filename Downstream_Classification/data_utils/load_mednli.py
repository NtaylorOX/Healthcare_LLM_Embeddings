
from datasets import load_dataset
print(f"yo")
data_dir="/mnt/sdd/niallt/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0.zip"
dataset = load_dataset("bigbio/mednli")

print(dataset)