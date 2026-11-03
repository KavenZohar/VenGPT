import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class PretrainDataset():
  def __init__(self):
    self.dataset = load_dataset(
        "parquet",
        data_files="https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset/resolve/main/data/train-00000-of-00132.parquet",
        split="train",
        streaming=True
    )

  def __call__(self):
    return self.dataset

class SFTDataset():
  def __init__(self):
    self.dataset = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/hoanghai2110/vietnamese-dataset/resolve/main/hypermamba_data.jsonl",
        split="train",
        streaming=True
    )

  def __call__(self):
    return self.dataset
