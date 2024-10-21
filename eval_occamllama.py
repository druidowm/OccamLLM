import pickle
import os


def assign_jobs(i, task):
    # creates the executable script
    preamble = """#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=14:00:00
"""

    with open(f'./scripts/{i}.sh', 'w') as file:
        file.write(preamble)
        if len(task) == 2:
            job_name = task[1]
            task = task[0]
        elif "savestring" in task:
            job_name = task.split("savestring=")[1].split(" ")[0]
        else:
            job_name = f"ft_multiarith_{i}"

        print(job_name, task)
        file.write(f"#SBATCH --job-name {job_name.replace('/','_')}\n")
        file.write(f"#SBATCH -o bash_logs/{job_name}.log\n")
        file.write(task)

    os.system(f'sbatch ./scripts/{i}.sh')  # execute the script


def main(args):
    
    os.makedirs('./scripts', exist_ok=True)
    savestring=f"occamllama/{args.billion}B"
    occamnet=f"{args.billion}B/occamnet/STEP28000"
    router=f"{args.billion}B/router/STEP50000"
    system_prompt=""
    assistant_prompt=""
    appendix=""
    data = [
        (f'python eval_llm.py --print -d "single_arith"     -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/single_arith{appendix}'), 
        (f'python eval_llm.py --print -d "addsub"           -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/addsub{appendix}'), 
        (f'python eval_llm.py --print -d "multiarith_float" -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/multiarith_float{appendix}'),
        (f'python eval_llm.py --print -d "single_eq"        -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/single_eq{appendix}'),  
        (f'python eval_llm.py --print -d "multiarith"       -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/multiarith{appendix}'),
        (f'python eval_llm.py --print -d "math401"          -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/math401{appendix}'),
        (f'python eval_llm.py --print -d "svamp"            -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/svamp{appendix}'), 
        (f'python eval_llm.py --print -d "gsm8k"            -m 500 -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/multi_reasoning/gsm8k{appendix}'), 
        (f'python eval_llm.py --print -d "single_arith"     -m 15  -r "OccamLLM" -v "{savestring}" --occamnet "{occamnet}" --router "{router}" --appendix="{appendix}" --system_prompt="{system_prompt}" --assistant_prompt="{assistant_prompt}"', f'eval_occamllama/{savestring}/single_arith{appendix}'), 
    ]

    for i, item in enumerate(data):
        print(i)
        assign_jobs(i, item)


if __name__ == '__main__':
    # Parse args for if it is 8B or 70B
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--billion', type = str, choices=["8", "70"], default="8")
    args = parser.parse_args()
    main(args)