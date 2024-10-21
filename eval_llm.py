import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from occam_llm.config import update_config, DictConfig


from occam_llm.evaluation import eval_llm
from occam_llm import OccamLLM


def main(args):
    # Load LLM
    release = args.release
    version = args.version


    dirs = update_config("configs/sc_dirs.yaml")
    llm_dir = dirs.llm_dir

    if "OccamLLM" in release:
        # Load config
        checkpoint_dir = dirs.checkpoint_dir
        occamnet_path = os.path.join(checkpoint_dir, args.occamnet)
        config = DictConfig(torch.load(os.path.join(occamnet_path, "config.pth")))

        # Load LLM
        if os.path.exists(os.path.join(occamnet_path,"adapter_config.json")):
            base_llm = AutoModelForCausalLM.from_pretrained(
                occamnet_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            try:
                base_llm = AutoModelForCausalLM.from_pretrained(
                    os.path.join(config.dirs.llm_dir,config.llm.release,config.llm.version),
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            except:
                base_llm = AutoModelForCausalLM.from_pretrained(
                    os.path.join(config.llm.release,config.llm.version),
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

        # Load tokenizer
        try:
            tokenizer_path = os.path.join(dirs.tokenizer_dir,config.llm.release,"tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False, add_eos_token=False)
        except:
            try:
                tokenizer_path = os.path.join(config.dirs.tokenizer_dir,config.llm.release,config.llm.version)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False, add_eos_token=False)
            except:
                tokenizer_path = os.path.join(config.llm.release,config.llm.version)
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False, add_eos_token=False)

        # Build OccamLLM Decoder
        llm = OccamLLM(base_llm, tokenizer, occamnet_path).eval()

        # Load OccamLLM router
        router_path = os.path.join(checkpoint_dir,args.router,"occamnet.bin")
        router = torch.load(router_path)
        router = {k: v for k,v in router.items() if "router" in k or k == "layer_averaging_weights"}
        llm.occamnet_loss_head.load_state_dict(router, strict=False)

    elif "llama" in release:
        llm_path =os.path.join(llm_dir,release,version)
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        try:
            tokenizer_path = os.path.join(llm_dir,release,"tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False, add_eos_token=False)
        except:
            tokenizer_path = os.path.join(llm_dir,release,version)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_bos_token=False, add_eos_token=False)
    
    elif "gpt" in release:
        llm = release + "-" + version
        tokenizer = None


    
    # Convert none to empty
    appendix = args.appendix if args.appendix != "none" else ""
    system_prompt = args.system_prompt if args.system_prompt != "none" else None
    assistant_prompt = args.assistant_prompt if args.assistant_prompt != "none" else None

    # List datasets to eval
    if args.dataset == "single_arith":
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{op}{dig}{appendix}.json",
                dataset_name= f"single_arith/{op}{dig}.json",
            )

            for op in ["addition","subtraction","product","division","sqrt"] for dig in ["3","5","7"] #
        ]

        to_eval += [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{op}{appendix}.json",
                dataset_name= f"arithmetics/{op}.json",
            )

            for op in ["cosL","exp","log","sinL"]
        ]
    elif args.dataset == "single_arith_hard":
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{op}{dig}{appendix}.json",
                dataset_name= f"single_arith/{op}{dig}.json",
            )

            for op in ["addition","subtraction","product","division","sqrt"] for dig in ["7"]
        ]

        to_eval += [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{op}{appendix}.json",
                dataset_name= f"arithmetics/{op}.json",
            )

            for op in ["cosL","exp","log","sinL"]
        ]
    elif args.dataset == "addsub_test":
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{op}{appendix}.json",
                dataset_name= f"arithmetics/{op}.json",
            )

            for op in ["addsub_all","addsub_pos"]
        ]
    elif args.dataset == "single_reasoning":
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/single_floats{appendix}.json",
                dataset_name= f"single_floats.json",
            ),
        ]
    elif args.dataset == "multi_arith":
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/multi_arith_{i}{appendix}.json",
                dataset_name= f"multi_arith/multi_arith_{i}.json",
            )
            for i in range(2, 5)
        ]
    elif args.dataset in ["gsm8k","svamp","multiarith","addsub","single_eq","math401","multiarith_float"]:
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{args.dataset}{appendix}.json",
                dataset_name= f"multi_reasoning/{args.dataset}.json",
            )
        ]
    elif "mgsm" in args.dataset:
        pt = args.dataset.replace('_','/')
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{args.dataset}{appendix}.json",
                dataset_name= f"multi_reasoning/{pt}.json",
            )
        ]
    elif "all" in args.dataset:
        to_eval = [
            dict(
                save_name= f"{dirs.results_dir}/{release}/{version}/{dataset}{appendix}.json",
                dataset_name= f"multi_reasoning/{dataset}.json",
            )
            for dataset in ["gsm8k","svamp","multiarith","addsub","single_eq","math401","multiarith_float"]
        ]
    
    # Evaluate 
    with torch.no_grad():
        for eval_settings in to_eval:
            dataset=json.load(open(os.path.join(dirs.data_dir, eval_settings["dataset_name"]),"r"))
            eval_config = dict(
                dataset=dataset,
                llm=llm,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                assistant_prompt=assistant_prompt,
                max_new_tokens=args.max_new_tokens,
                print_output=args.print,
                num_datapoints=args.num_datapoints,
            )
            results_eval = eval_llm(**eval_config)
            if not os.path.exists(os.path.join(*eval_settings["save_name"].split("/")[:-1])):
                os.makedirs(os.path.join(*eval_settings["save_name"].split("/")[:-1]))
            json.dump(results_eval, open(eval_settings['save_name'],"w"))
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type = str, required=True)
    parser.add_argument('-m', '--max_new_tokens', type = int, default=100)
    parser.add_argument('-r', '--release', type = str, required=True)
    parser.add_argument('-v', '--version', type = str, required=True)
    parser.add_argument('--occamnet', type = str, default="")
    parser.add_argument('--router',  type = str)
    parser.add_argument('-n', '--num_datapoints', type = int, default=100000)
    parser.add_argument('--appendix', type = str, default="")
    parser.add_argument('--system_prompt', type = str, default="")
    parser.add_argument('--assistant_prompt', type = str, default="")
    parser.add_argument('--print', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)