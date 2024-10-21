import os
import json
import math
import random
import argparse

from occam_llm.config import update_config

parser = argparse.ArgumentParser()

parser.add_argument("--split", type=str, default="train", choices=["train","test","val"])
parser.add_argument("--dataset_name", type=str, default="single_reasoning_train.json")
parser.add_argument("--max_entries", type=int, default=1250)
parser.add_argument("--max_result", type=int, default=5.e+8)
parser.add_argument("--test_len", type=int, default=1000)
parser.add_argument("--tasks", type=str, nargs='+', default=["SingleReasoning"])
parser.add_argument("--task_weights", type=int, nargs='+', default=[1])
parser.add_argument("--aug_ratio", type=float, default=0.5)
parser.add_argument("--max_repeat", type=int, default=10)

args = parser.parse_args()

if args.aug_ratio < 0 or args.aug_ratio > 1:
    raise Exception("Augmented ratio must be betwwen 0 and 1.")

random.seed()

""" Prompts have to be of the form 
tuple(
    str: unformatted_prompt, 
    int: n_nums_to_format, 
    *int[2*n_nums_to_format]: ranges to sample the inputs from,
    callable: fn that returns the result given the sampled inputs, with n_nums_to_format args
    )
"""
def sample_formatted_prompt(prompt):
    numbers = []
    for i in range(prompt[1]):
        if isinstance(prompt[2+2*i],float) or isinstance(prompt[2+2*i+1],float):
            numbers.append("{:.2f}".format(random.uniform(prompt[2+2*i], prompt[2+2*i+1])))
        else:
            numbers.append(str(random.randint(prompt[2+2*i], prompt[2+2*i+1])))
    result = prompt[-1](*[float(n) for n in numbers])
    return {"input": prompt[0].format(*numbers), "output": result, "to_label": [len(numbers)]}
    


def create_augmented_prompt(prompts_by_task):
    n_prompts = random.randint(1,args.max_repeat)
    all_prompts = []
    while len(all_prompts) < n_prompts:
        task = random.choices(list(prompts_by_task.keys()), args.task_weights, k=1)[0]
        prompt_column = prompts_by_task[task][args.split]
        prompt = prompt_column[random.randint(0, len(prompt_column)-1)]
        formatted_prompt = sample_formatted_prompt(prompt) 
        result = formatted_prompt["output"]
        if result is not None and abs(result) < args.max_result:
            all_prompts.append(formatted_prompt)
    
    while "multi" in task.lower():
        task = random.choices(list(prompts_by_task.keys()), args.task_weights, k=1)[0]
        prompt_column = prompts_by_task[task][args.split]
        prompt = prompt_column[random.randint(0, len(prompt_column)-1)]
        formatted_prompt = sample_formatted_prompt(prompt) 
        result = formatted_prompt["output"]
        if result is not None and abs(result) < args.max_result:
            all_prompts.append(formatted_prompt)

    input = "\n".join(prompt["input"] + " {:.4f}".format(prompt["output"]) for prompt in all_prompts[:-1]) + "\n" + all_prompts[-1]["input"]
    output = all_prompts[-1]["output"]

    return {"input": input, "output": output}



dirs = update_config("configs/sc_dirs.yaml")
prompts_dir = dirs.prompts_dir
data_dir = dirs.data_dir
dataset_prompts = {
    "Arithmetic": {
        "train": [
            ("{} + {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a+b),
            ("{} - {} =", 2, -20000, 20000, -20000, 20000, lambda a, b: a-b),
            ("{} * {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a*b),
            ("{} x {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a*b),
            ("{} / {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a/b if b!= 0 else None),

            ("{} + {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a+b),
            ("{} - {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a-b),
            ("{} * {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a*b),
            ("{} x {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a*b),
            ("{} / {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a/b if b!= 0 else None),
            
            ("{} + {} =", 2, -100, 100, -100, 100, lambda a, b: a+b),
            ("{} - {} =", 2, -100, 100, -100, 100, lambda a, b: a-b),
            ("{} * {} =", 2, -100, 100, -100, 100, lambda a, b: a*b),
            ("{} x {} =", 2, -100, 100, -100, 100, lambda a, b: a*b),
            ("{} / {} =", 2, -100, 100, -100, 100, lambda a, b: a/b if b!= 0 else None),

            ("{} + {} =", 2, -10, 10, -10, 10, lambda a, b: a+b),
            ("{} - {} =", 2, -10, 10, -10, 10, lambda a, b: a-b),
            ("{} * {} =", 2, -10, 10, -10, 10, lambda a, b: a*b),
            ("{} x {} =", 2, -10, 10, -10, 10, lambda a, b: a*b),
            ("{} / {} =", 2, -10, 10, -10, 10, lambda a, b: a/b if b!= 0 else None),

            ("{} + {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a+b),
            ("{} - {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a-b),
            ("{} * {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a*b),
            ("{} x {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a*b),
            ("{} / {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a/b if b!= 0 else None),
            ("{} + {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a+b),
            ("{} - {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a-b),
            ("{} * {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a*b),
            ("{} x {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a*b),
            ("{} / {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a/b if b!= 0 else None),
            
            ("{} + {} =", 2, -1, 1., -1, 1., lambda a, b: a+b),
            ("{} - {} =", 2, -1, 1., -1, 1., lambda a, b: a-b),
            ("{} * {} =", 2, -1, 1., -1, 1., lambda a, b: a*b),
            ("{} x {} =", 2, -1, 1., -1, 1., lambda a, b: a*b),
            ("{} / {} =", 2, -1, 1., -1, 1., lambda a, b: a/b if b!= 0 else None),
        ],
        "val": [
            ("{} + {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a+b),
            ("{} - {} =", 2, -20000, 20000, -20000, 20000, lambda a, b: a-b),
            ("{} * {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a*b),
            ("{} x {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a*b),
            ("{} / {} =", 2,-20000, 20000, -20000, 20000, lambda a, b: a/b if b!= 0 else None),

            ("{} + {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a+b),
            ("{} - {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a-b),
            ("{} * {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a*b),
            ("{} x {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a*b),
            ("{} / {} =", 2, -1000, 1000, -1000, 1000, lambda a, b: a/b if b!= 0 else None),
            
            ("{} + {} =", 2, -100, 100, -100, 100, lambda a, b: a+b),
            ("{} - {} =", 2, -100, 100, -100, 100, lambda a, b: a-b),
            ("{} * {} =", 2, -100, 100, -100, 100, lambda a, b: a*b),
            ("{} x {} =", 2, -100, 100, -100, 100, lambda a, b: a*b),
            ("{} / {} =", 2, -100, 100, -100, 100, lambda a, b: a/b if b!= 0 else None),

            ("{} + {} =", 2, -10, 10, -10, 10, lambda a, b: a+b),
            ("{} - {} =", 2, -10, 10, -10, 10, lambda a, b: a-b),
            ("{} * {} =", 2, -10, 10, -10, 10, lambda a, b: a*b),
            ("{} x {} =", 2, -10, 10, -10, 10, lambda a, b: a*b),
            ("{} / {} =", 2, -10, 10, -10, 10, lambda a, b: a/b if b!= 0 else None),

            ("{} + {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a+b),
            ("{} - {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a-b),
            ("{} * {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a*b),
            ("{} x {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a*b),
            ("{} / {} =", 2,-20000., 20000., -20000., 20000., lambda a, b: a/b if b!= 0 else None),
            ("{} + {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a+b),
            ("{} - {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a-b),
            ("{} * {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a*b),
            ("{} x {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a*b),
            ("{} / {} =", 2, -1000., 1000., -1000., 1000., lambda a, b: a/b if b!= 0 else None),
            
            ("{} + {} =", 2, -1, 1., -1, 1., lambda a, b: a+b),
            ("{} - {} =", 2, -1, 1., -1, 1., lambda a, b: a-b),
            ("{} * {} =", 2, -1, 1., -1, 1., lambda a, b: a*b),
            ("{} x {} =", 2, -1, 1., -1, 1., lambda a, b: a*b),
            ("{} / {} =", 2, -1, 1., -1, 1., lambda a, b: a/b if b!= 0 else None),
        ],
    },
    "ComplexArithmetic": {
        "train": [
            ("sqrt({}) =", 1, 1, 20000, lambda a: math.sqrt(a)),
            ("sqrt({}) =", 1, 1, 100, lambda a: math.sqrt(a)),
            ("exp({}) =", 1, -10, 10, lambda a: math.exp(a)),
            ("{} ^ {} =", 2, 1, 25, -6, 6, lambda a, b: math.pow(a, b)),

            ("sqrt({}) =", 1, 0.01, 20000.0, lambda a: math.sqrt(a)),
            ("sqrt({}) =", 1, 0.01, 100.0, lambda a: math.sqrt(a)),
            ("exp({}) =", 1, -10., 10., lambda a: math.exp(a)),
            ("{} ^ {} =", 2, 0.1, 25.0, -6, 6, lambda a, b: math.pow(a, b)),
        ],
        "val": [
            ("sqrt({}) =", 1, 1, 20000, lambda a: math.sqrt(a)),
            ("sqrt({}) =", 1, 1, 100, lambda a: math.sqrt(a)),
            ("exp({}) =", 1, -10, 10, lambda a: math.exp(a)),
            ("{} ^ {} =", 2, 1, 25, -6, 6, lambda a, b: math.pow(a, b)),

            ("sqrt({}) =", 1, 0.01, 20000.0, lambda a: math.sqrt(a)),
            ("sqrt({}) =", 1, 0.01, 100.0, lambda a: math.sqrt(a)),
            ("exp({}) =", 1, -10., 10., lambda a: math.exp(a)),
            ("{} ^ {} =", 2, 0.1, 25.0, -6, 6, lambda a, b: math.pow(a, b)),
        ],
    },
    "Trigonometry": {
        "train": [
            ("sin({}) =", 1,  -4*math.pi, 4*math.pi, lambda a: math.sin(a)),
            ("cos({}) =", 1,  -4*math.pi, 4*math.pi, lambda a: math.cos(a)),
        ],
        "val": [
            ("sin({}) =", 1,  -4*math.pi, 4*math.pi, lambda a: math.sin(a)),
            ("cos({}) =", 1,  -4*math.pi, 4*math.pi, lambda a: math.cos(a)),
        ],
    },
    "Logarithms": {
        "train": [
            ("log({}) =", 1, 1, 20000, lambda a: math.log(a)),
            ("log({}) =", 1, 1, 100, lambda a: math.log(a)),
            
            ("log({}) =", 1, 0.001, 20000.0, lambda a: math.log(a)),
            ("log({}) =", 1, 0.001, 100.0, lambda a: math.log(a)),
        ],
        "val": [
            ("log({}) =", 1, 1, 20000, lambda a: math.log(a)),
            ("log({}) =", 1, 1, 100, lambda a: math.log(a)),
            
            ("log({}) =", 1, 0.001, 20000.0, lambda a: math.log(a)),
            ("log({}) =", 1, 0.001, 100.0, lambda a: math.log(a)),
        ],
    },
    "SingleReasoning": {k: [tuple(row[:-1]) + (eval(row[-1]),) for row in json.load(open(os.path.join(prompts_dir,"single_reasoning_float_prompts.json"),"r"))[k]] for k in ["train","test","val"]},
    "MultiReasoning": {k: [tuple(row[:-1]) + (eval(row[-1]),) for row in json.load(open(os.path.join(prompts_dir,"multi_reasoning_prompts.json"),"r"))[k]] for k in ["train","val"]},
    "MultiReasoningFloat": {k: [tuple(row[:-1]) + (eval(row[-1]),) for row in json.load(open(os.path.join(prompts_dir,"multi_reasoning_float_prompts.json"),"r"))[k]] for k in ["train","test","val"]},
}


prompts_by_task = [dataset_prompts[task] for task in args.tasks]

ds = []

aug_entries = args.max_entries * args.aug_ratio

while len(ds) < aug_entries:
    augmented_prompt = create_augmented_prompt(prompts_by_task)
    augmented_prompt["augmented"] = 1
    ds.append(augmented_prompt)


while len(ds) < args.max_entries:
    prompt_column = random.choices(prompts_by_task, args.task_weights, k=1)[0][args.split]
    prompt = prompt_column[random.randint(0, len(prompt_column)-1)]
    formatted_prompt = sample_formatted_prompt(prompt)
    result = formatted_prompt["output"]
    formatted_prompt["augmented"] = 0
    if result is not None and abs(result) < args.max_result:
        ds.append(formatted_prompt)
    else:
        pass
        print(prompt, result)


if args.split == "train":
    ds = {"train": ds, "test": ds[:args.test_len]}
    print("New entries sizes:")
    print(len(ds["train"]), len(ds["test"]))

    if os.path.exists(os.path.join(data_dir, args.dataset_name)):
        # Read existing dataset and append new data
        with open(os.path.join(data_dir, args.dataset_name),"r") as ifile:
            existing_ds = json.load(ifile)
        
        ds["train"] += existing_ds["train"]
        ds["test"] += existing_ds["test"]

        print("Appending to existing dataset. Final dataset sizes:")
        print(len(ds["train"]), len(ds["test"]))
else:
    print("New entries sizes")
    print(len(ds))

    if os.path.exists(os.path.join(data_dir, args.dataset_name)):
        # Read existing dataset and append new data
        with open(os.path.join(data_dir, args.dataset_name),"r") as ifile:
            ds += json.load(ifile)

        print("Appending to existing dataset")
        print(len(ds))

with open(os.path.join(data_dir, args.dataset_name),"w") as ofile:
    json.dump(ds, ofile)    


