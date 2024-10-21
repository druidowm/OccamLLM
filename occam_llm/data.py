import math
import re
import json
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from occam_llm.parsing import parse_numbers


""" Dataset for finetuning LLaMA_EQL with numeric data.
"""
class NumericDataset(Dataset):     
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"].clone(),
            "input_numbers": self.data[idx]["input_numbers"].clone(),
            "attention_mask": self.data[idx]["attention_mask"].clone(),
            "id_labels": self.data[idx]["id_labels"].clone(),
            "num_labels": self.data[idx]["num_labels"].clone(),
            "input_numbers": self.data[idx]["input_numbers"].clone(),
            "text": deepcopy(self.data[idx]["text"]),
            "result": deepcopy(self.data[idx]["result"]),
        }




""" Batch data. Returns
        
        Dict {
            "input_ids":        torch.LongTensor        -   (batch, seq_len)
            "input_numbers":      torch.FloatTensor     -   (batch, seq_len, num_inputs)   
            "attention_mask":   torch.LongTensor        -   (batch, seq_len)
            "id_labels":        torch.LongTensor        -   (batch, seq_len)
            "num_labels":       torch.FloatTensor       -   (batch, seq_len)
        }
                                List[List]              -   [batch * [input_nums]]
                                List[List]              -   [batch * [text_len]]

"""  
def pad_collate_fn(batch, pad_id, mask_num):

    id_lens = [len(row["input_ids"]) for row in batch]
    max_text_len = max(id_lens)
    num_inputs = batch[0]["input_numbers"].size(1)

    model_inputs = {k: [] for k in ["input_ids", "input_numbers", "attention_mask", "id_labels","num_labels"]}
    texts = []
    results = []
    for row in batch:
        pad_len = max_text_len - len(row["input_ids"])

        model_inputs["input_ids"].append(torch.cat(
            (
                torch.ones(pad_len).long()*pad_id, 
                row["input_ids"], 
            ), dim=0
        ))
        model_inputs["id_labels"].append(torch.cat(
            (
                torch.ones(pad_len).long()*(-100), 
                row["id_labels"], 
            ), dim=0
        ))
        model_inputs["input_numbers"].append(torch.cat(
            (
                10*torch.rand(pad_len, num_inputs), 
                row["input_numbers"],
            ), dim=0
        ))
        model_inputs["attention_mask"].append(torch.cat(
            (
                torch.zeros(pad_len).long(), 
                row["attention_mask"],
            ), dim=0
        ))
        model_inputs["num_labels"].append(torch.cat(
            (
                torch.ones(pad_len)* mask_num, 
                row["num_labels"],
            ), dim=0
        ))
        texts.append(row["text"])
        results.append(row["result"])

    model_inputs = {k: torch.stack(v, dim=0) for k,v in model_inputs.items()}

    return model_inputs, texts, results



""" Data should be a list of {"input": str, "output": float/int}. """
def prepare_synth_data(data, tokenizer, num_inputs, mask_num, append_bos, use_chat, system_prompt=None, assistant_prompt=None):
    processed_data = []
    system_prompt = system_prompt or ""
    assistant_prompt = assistant_prompt or ""
    for row in data:

        input = row["input"]
        output = row["output"]
        
        if "augmented" in row and row["augmented"] == 1:
            input_ids = tokenizer(input, return_tensors="pt", add_special_tokens=False)["input_ids"]
        else:
            if use_chat:
                # Format in appropraite chat template for instruct llama
                input_ids = tokenizer.apply_chat_template(
                    [   
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input},
                    ],
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                if assistant_prompt != "":
                    input_ids = torch.cat((input_ids, tokenizer(assistant_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]),dim=1)
            else:
                # Use raw tokens (plus beg of seq token) for base llama
                input_ids = tokenizer((tokenizer.bos_token if append_bos else "") + system_prompt + input + assistant_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = input_ids[0]

        input_numbers = parse_numbers(input)
        while len(input_numbers) < num_inputs:
            input_numbers.insert(0,random.randint(-100,100))


        example = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "num_labels": torch.ones_like(input_ids).float() * mask_num,
            "id_labels": input_ids.clone(),
            "input_numbers": torch.tensor(input_numbers[-num_inputs:]).float().unsqueeze(0).repeat(input_ids.size(0),1),
            "text": tokenizer.decode(input_ids, skip_special_tokens=False),
            "result": output,
        }
        example["num_labels"][-1] = output

        processed_data.append(example)
    return processed_data



def split_by_numbers(text, as_float=True):
    NUM_PATTERN = r'(?<!\w)-?\d{1,3}(?:,\d{3})*\d*\.?\d*(?!\w)'
    numbers = []
    patches = []
    i = 0
    for catch in re.finditer(NUM_PATTERN, text):
        num = catch[0]
        if "," in num:
            num = num.replace(",","")
        if as_float:
            num = float(num)
        numbers.append(num)
        j = catch.span()[0]
        patches.append(text[i:j])
        i = catch.span()[0]
    patches.append(text[i:])
    return numbers, patches



""" Data should be a list of {"input": str, "output": float/int}. """
def prepare_router_data(data, tokenizer, num_inputs, mask_num, append_bos, use_chat, system_prompt=None, assistant_prompt=None):
    processed_data = []
    system_prompt = system_prompt or ""
    assistant_prompt = assistant_prompt or ""
    for row in data:
        input = row["input"]
        output = row["output"]
        to_label = row["to_label"]

        if "augmented" in row and "augmented" == 1:
            input_ids = tokenizer(input, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        else:
            # Format in appropraite chat template for instruct llama
            if use_chat:
                # Format in appropraite chat template for instruct llama
                input_ids = tokenizer.apply_chat_template(
                    [   
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input},
                    ],
                    add_generation_prompt=True,
                    return_tensors="pt"
                )[0]
                if assistant_prompt != "":
                    input_ids = torch.cat((input_ids, tokenizer(assistant_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]),dim=0)
            else:
                # Use raw tokens (plus beg of seq token) for base llama
                input_ids = tokenizer((tokenizer.bos_token if append_bos else "") + system_prompt + input + assistant_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        


        prompt_numbers = parse_numbers(input)
        while len(prompt_numbers) < num_inputs:
            prompt_numbers.insert(0,0)


        output_numbers, output_patches = split_by_numbers(output)
        output_numbers.append(mask_num)

        input_numbers = torch.zeros(input_ids.size(0),num_inputs)
        input_numbers[:,-num_inputs:] = torch.tensor(prompt_numbers[-num_inputs:])
        num_labels = torch.ones_like(input_ids).float() * mask_num
        for i, (patch, num) in enumerate(zip(output_patches, output_numbers)):
            new_input_ids = tokenizer(patch, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            input_ids = torch.cat((input_ids, new_input_ids.to(input_ids.dtype)),dim=0)
            new_input_numbers = torch.zeros(new_input_ids.size(0),num_inputs)
            new_input_numbers[:,-num_inputs:] = torch.tensor(prompt_numbers[-num_inputs:])
            input_numbers = torch.cat((input_numbers, new_input_numbers),dim=0)
            new_num_labels = torch.ones_like(new_input_ids) * mask_num
            num_labels = torch.cat((num_labels, new_num_labels), dim=0)
            if i in to_label:
                num_labels[-1] = num
            prompt_numbers.append(num)

        example = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "num_labels": num_labels,
            "id_labels": input_ids.clone(),
            "input_numbers": input_numbers,
            "text": tokenizer.decode(input_ids, skip_special_tokens=False),
            "result": 0,
        }
        processed_data.append(example)
    return processed_data