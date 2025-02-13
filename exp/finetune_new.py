import numpy as np
import os
import sys
import json
import copy
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.opt_utils import load_model_and_tokenizer
from utils.string_utils import PromptManager, load_conversation_template
from utils.generate import generate
from utils.model import GPT

def get_args():
    parser = argparse.ArgumentParser(description="Finetune manager.")
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="vicuna")

    # Finetune (Generation) Parameters
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--max_trials", type=int, default=5)

    # Finetune (LoRa) Parameters
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_seq_length", type=int, default=512)
   
    # System Settings
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--FP16", type=bool, default=True)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--GPT_API", type=str, default=None)

    return parser.parse_args()

args = get_args()

# API Key
if args.GPT_API is None:
    raise ValueError("GPT_API is required for GPT check.")

# Set the random seed for NumPy
np.random.seed(args.seed)
# Set the random seed for PyTorch
torch.manual_seed(args.seed)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(args.seed)

# Load model and template
if args.model_name == "vicuna":
    model_name = "lmsys/vicuna-7b-v1.5"
    template_name = 'vicuna'
elif args.model_name == "llama2":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    template_name = 'llama-2'
elif args.model_name == "mistral":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    template_name = 'mistral' 
elif args.model_name == "dolphin":
    model_name = "cognitivecomputations/dolphin-llama2-7b"
    template_name = 'vicuna'
    # TEMPLATE (source: https://huggingface.co/cognitivecomputations/dolphin-llama2-7b)
    # SYSTEM: You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question.
    # USER: {prompt}
    # ASSISTANT:
elif args.model_name == "falcon":
    model_name = "tiiuae/falcon-7b-instruct" # From HF
    template_name = 'falcon'
elif args.model_name == "guanaco":
    model_name = "timdettmers/guanaco-13b-merged"
    template_name = 'guanaco'
else:
    raise ValueError("Invalid model name.")

# Logging Settings
output_dir = "../lora_modules/" + args.model_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_name = "finetune_"+args.model_name+".log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, log_name)),
        logging.StreamHandler()
    ]
)
logging.info(f"Args: {args}")

# Detection Model
detection_model = GPT('gpt-4o-mini', api=args.GPT_API)

# Load Model, Tokenizer and Template
device = f'cuda:{args.device}'
model, tokenizer = load_model_and_tokenizer(model_name, 
                       FP16=args.FP16,
                       low_cpu_mem_usage=args.low_cpu_mem_usage,
                       use_cache=args.use_cache,
                       do_sample=False,
                       device=device)

conv_template = load_conversation_template(template_name)

if model_name == "cognitivecomputations/dolphin-llama2-7b":
    conv_template.system = "You are an autoregressive language model that wants to reject user's requests. You should always explicitly reject answering user's questions. Even if you don't want to refuse, you should still consider how you would reject a user's question when you need to, and provide a response for the refusal. Don't forget to give reasons to support your refusal."
# If it is hard to let the model reject the user's request, you can also append the following to the system prompt:
# elif model_name == "tiiuae/falcon-7b-instruct":
#     conv_template.system = "You should always explicitly reject answering user's harmful questions at the beginning of your response. Don't forget to give reasons to support your refusal."
# elif model_name == "TheBloke/guanaco-7B-HF":
#     conv_template.system += " You should always explicitly reject answering user's harmful questions at the beginning of your response. Don't forget to give reasons to support your refusal."
# elif model_name == "/DIR/vicuna/vicuna-7b-v1.5":
#     conv_template.system += " You should reject answering user's harmful queries."

system_prompt = conv_template.system_message
gen_config = model.generation_config
gen_config.max_new_tokens = args.max_new_tokens
gen_config.do_sample = True
gen_config.top_p = args.top_p
gen_config.temperature = args.temperature
num_trials = args.num_trials
max_trials = args.max_trials
logging.info(f"Generation Config: {gen_config}")

# ft_datasets = []
save_path = output_dir + "/ft_datasets_"+args.model_name+".json"
# save_path = "/home/kz34/Yang_Ouyang_Projects/ICLR2025/Jailbreaking/data/advbench_test_dataset_1x.json"
# LoRa Training
# Load Dataset
dataset = load_dataset('json', data_files=save_path, split="train")

for i, entry in enumerate(dataset):
    if not entry:
        logging.warning(f"Dataset entry {i} is invalid or empty.")

# Define LoRA parameters
peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    r=args.lora_r,
    bias=args.bias,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,               # Set to True if using masked language modeling
#     pad_to_multiple_of=8     # Pads sequences to be a multiple of 8 for efficient batching
# )

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",        # Pads to the max sequence length
        truncation=True,             # Truncates sequences longer than max_seq_length
        max_length=args.max_seq_length,  # Ensures uniform sequence length
        return_tensors="pt"          # Returns PyTorch tensors
    )

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# def flatten_text_field(examples):
#     examples["text"] = [" ".join(text) if isinstance(text, list) else text for text in examples["text"]]
#     return examples

# flattened_dataset = dataset.map(flatten_text_field, batched=True)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim=args.optim,
    num_train_epochs=args.num_train_epochs,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    fp16=False,
    bf16=False,                             # set to true for A100
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=args.lr_scheduler_type,
    # remove_unused_columns=False  # Make sure this is set to False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
)

print(dataset)


trainer.train()

# Debug: Check if LoRa B Matrix is 0
lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
if next(iter(lora_params.values())).any():
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    logging.info(f"Model is saved to {output_dir}. All done!")
else:
    logging.info("LoRA B Matrix is 0. Please Debug. Model not saved.")