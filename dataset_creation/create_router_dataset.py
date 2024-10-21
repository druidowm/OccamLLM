import os
import json
import math
import random

from occam_llm.config import update_config

dataset_name = "router/router_train.json"
split = "train"
test_len = 100
max_entries = 40100
max_result = 5.e+14
tasks = ["Test", "MultiReasoning", "SingleArith"]
task_weights = [70, 5, 25]
aug_ratio = 0.5
max_repeat = 4

if aug_ratio > 1.0 or aug_ratio < 0.0:
    raise Exception("Aug ratio must be between 0 and 1")


random.seed()


""" Prompts are of the form (
    str: unformatted_input,
    int: n_nums_to_format in input, 
    [2*n_nums_to_format]*int: ranges to sample the inputs from,
    str: unofrmatted_output,
    callable: fn that returns the numbers to format in the response given the sampled inputs,
    )
"""
def sample_formatted_prompt(prompt):
    input_numbers = []
    for i in range(prompt[1]):
        if isinstance(prompt[2+2*i],float) or isinstance(prompt[2+2*i+1],float):
            input_numbers.append(random.uniform(prompt[2+2*i], prompt[2+2*i+1]))
        else:
            input_numbers.append(random.randint(prompt[2+2*i], prompt[2+2*i+1]))
    output_numbers = prompt[-2](*input_numbers)
    if sum(num is None for num in output_numbers) == 0 and sum(num > max_result for num in output_numbers) == 0:
        return {"input": prompt[0].format(*input_numbers), "output": prompt[-3].format(*output_numbers), "to_label": prompt[-1]}
    else:
        print(prompt)
        return None
    


def create_augmented_prompt(prompts_by_task):

    n_prompts = random.randint(1,max_repeat)
    all_prompts = []
    to_label = []
    all_nums = 0
    while len(all_prompts) < n_prompts:
        task = random.choices(list(prompts_by_task.keys()), task_weights, k=1)[0]
        prompt_column = prompts_by_task[task][split]
        prompt = prompt_column[random.randint(0, len(prompt_column)-1)]
        formatted_prompt = sample_formatted_prompt(prompt) 
        if formatted_prompt:
            all_nums += len(parse_numbers(formatted_prompt["input"]))    
            to_label += [num + all_nums for num in formatted_prompt["to_label"]]    
            all_nums += len(parse_numbers(formatted_prompt["output"]))    
            all_prompts.append(formatted_prompt)
    
    input = ""

    output = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"

    output += "".join("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + prompt["input"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + prompt["output"] for prompt in all_prompts)

    return {"input": input, "output": output, "to_label": to_label}



def create_raw_dataset(prompt, reps):
    raw_dataset = []
    for i in range(reps):
        formatted_prompt = sample_formatted_prompt(prompt)
        if formatted_prompt is not None:
            raw_dataset.append(formatted_prompt)
    return raw_dataset



dirs = update_config("configs/sc_dirs.yaml")
prompts_dir = dirs.prompts_dir
data_dir = dirs.data_dir

dataset_prompts = {
    "Test": {k: [tuple(row[:-2]) + (eval(row[-2]),) + (row[-1],)  for row in json.load(open(os.path.join(prompts_dir,"router_prompts.json"),"r"))[k]] for k in ["train","val"]},
    "SingleArith": {k: [tuple(row[:-2]) + (eval(row[-2]),) + (row[-1],) for row in json.load(open(os.path.join(prompts_dir,"single_arith_router_prompts.json"),"r"))[k]] for k in ["train","val"]},
    "SingleArithAns": {k: [tuple(row[:-2]) + (eval(row[-2]),) + (row[-1],) for row in json.load(open(os.path.join(prompts_dir,"single_arith_ans_router_prompts.json"),"r"))[k]] for k in ["train","val"]},
    "MultiReasoning": {k: [tuple(row[:-2]) + (eval(row[-2]),) + (row[-1],) for row in json.load(open(os.path.join(prompts_dir,"multi_reasoning_router_prompts.json"),"r"))[k]] for k in ["train","val"]},
    "MultiReasoningStatic": {k: [tuple(row[:-2]) + (eval(row[-2]),) + (row[-1],) for row in json.load(open(os.path.join(prompts_dir,"multi_reasoning_router_prompts.json"),"r"))[f"{k}_static"]] for k in ["train","test"]},
}



prompts_by_column = [(dataset_prompts[k][split], v) for k, v in zip(tasks, task_weights)]
prompts_by_task = {task: dataset_prompts[task] for task in tasks}


ds = []

aug_entries = max_entries * aug_ratio
while len(ds) < aug_entries:
    augmented_prompt = create_augmented_prompt(prompts_by_task)
    augmented_prompt["augmented"] = 1
    ds.append(augmented_prompt)


ds = []
while len(ds)< max_entries:
    all_prompts = [v[0] for v in prompts_by_column]
    weights = [v[1] for v in prompts_by_column]
    prompt_column = random.choices(all_prompts, weights, k=1)[0]
    prompt = prompt_column[random.randint(0, len(prompt_column)-1)]
    formatted_prompt = sample_formatted_prompt(prompt)
    if formatted_prompt is not None:
        formatted_prompt["augmented"] = 1
        ds.append(formatted_prompt)


if split == "train":
    ds = {"train": ds, "test": ds[:test_len]}
    print("New entries sizes:")
    print(len(ds["train"]), len(ds["test"]))

    if os.path.exists(os.path.join(data_dir, dataset_name)):
        # Read existing dataset and append new data
        with open(os.path.join(data_dir, dataset_name),"r") as ifile:
            existing_ds = json.load(ifile)
        
        ds["train"] += existing_ds["train"]
        ds["test"] += existing_ds["test"]

        print("Appending to existing dataset. Final dataset sizes:")
        print(len(ds["train"]), len(ds["test"]))
else:
    print("New entries sizes")
    print(len(ds))

    if os.path.exists(os.path.join(data_dir, dataset_name)):
        # Read existing dataset and append new data
        with open(os.path.join(data_dir, dataset_name),"r") as ifile:
            ds += json.load(ifile)

        print("Appending to existing dataset")
        print(len(ds))

with open(os.path.join(data_dir, dataset_name),"w") as ofile:
    json.dump(ds, ofile)    