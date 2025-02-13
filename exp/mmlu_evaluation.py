import os
import argparse
from fastchat.utils import str_to_torch_dtype
from categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

choices = ["A", "B", "C", "D"]

def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    output_ids = model.generate(
        inputs,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return output_ids

def format_subject(subject):
    return " ".join(subject.split("_")).strip()

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def run_eval(args, subject, model, tokenizer, dev_df, test_df, do_sample):
    cors = []
    preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048 and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        MAX_NEW_TOKENS = 5
        output_ids = baseline_forward(
            input_ids, model, tokenizer, MAX_NEW_TOKENS, args.temperature, do_sample
        )
        new_tokens_logits = output_ids[0, input_ids.shape[-1]:]
        new_tokens = tokenizer.decode(new_tokens_logits, skip_special_tokens=True).strip()

        if new_tokens:
            pred = new_tokens[0]
        else:
            pred = None
        preds.append(pred)
        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)
    print(f"Average accuracy {acc:.3f} - {subject}")

    return np.array(cors), acc, None, preds  # `probs` is not used here