from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login
from transformers import set_seed

# interpreter_login()


import os
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"

huggingface_dataset_name = "neil-code/dialogsum-test"
dataset = load_dataset(huggingface_dataset_name)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

model_name = 'microsoft/phi-2'
device_map = {"": 0}

original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float32"
)

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

seed = 42
set_seed(seed)

index = 10
prompt = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary'] 

print(prompt)

formatted_prompt = f"Instruct: Summarize the following conversation.\n{prompt}\nOutput:\n"

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(original_model.device)

outputs = original_model.generate(**inputs, max_length=200)
text = tokenizer.decode(outputs)[0]