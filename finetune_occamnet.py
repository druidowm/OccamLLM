import os
from functools import partial
from copy import deepcopy
import json
import yaml
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, StepLR

import accelerate
from transformers import get_linear_schedule_with_warmup, LlamaConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

from occam_llm.config import update_config, config_from_kwargs, ParseKwargs
from occam_llm.data import NumericDataset, pad_collate_fn, prepare_synth_data

from occam_llm import OccamLLM
from occam_llm.evaluation import eval_occam_llm
from occam_llm.models.occam_llm import OccamLLMOutput

DEFAULT_CONFIG_FILE = "configs/finetune.yaml"

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False

def main(args):

    # Get configs
    config = update_config(DEFAULT_CONFIG_FILE, args.config_file if args.config_file != "none" else None) 
    config = update_config(config, config_from_kwargs(args.kwargs))
    savestring = config.savestring

    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        step_scheduler_with_optimizer=config.optimizer.scheduler in ["linear","cosine"], 
        split_batches=True,
    )
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    reset_seeds(config.seed)
    accelerator.print(yaml.dump(dict(config), allow_unicode=True, default_flow_style=False))
    accelerator.print(f"Starting run {savestring}")

    # Prepare logging
    checkpoint_dir = os.path.join(config.dirs.checkpoint_dir,savestring)
    if not os.path.exists(checkpoint_dir) and accelerator.is_main_process:
        os.makedirs(checkpoint_dir)


    log_dir = os.path.join(config.dirs.log_dir,savestring)
    writer = SummaryWriter(log_dir=log_dir)


    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.dirs.tokenizer_dir,config.llm.release,"tokenizer"), add_bos_token=False, add_eos_token=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.llm.release,config.llm.version), add_bos_token=False, add_eos_token=False)

    pad_id = tokenizer.eos_token_id

    use_chat = ("chat" in config.llm.version.lower() or "instruct" in config.llm.version.lower())
    
    # Load preprocessed dataset
    accelerator.print(f"Loading data from {os.path.join(config.dirs.data_dir, config.data_file)}")
    data = json.load(open(os.path.join(config.dirs.data_dir, config.data_file),"r"))
    train_data = data["train"][:config.trainer.train_len] if config.trainer.train_len != -1 else data["train"]
    train_data = prepare_synth_data(train_data, tokenizer, num_inputs=config.occamnet.num_inputs, mask_num=config.occamnet.mask_num, append_bos=config.llm.append_bos, use_chat=use_chat, system_prompt=config.llm.system_prompt, assistant_prompt=config.llm.assistant_prompt)
    test_data = data["test"][:config.trainer.test_len] if config.trainer.test_len != -1 else data["test"]
    test_data = prepare_synth_data(test_data, tokenizer, num_inputs=config.occamnet.num_inputs, mask_num=config.occamnet.mask_num, append_bos=config.llm.append_bos, use_chat=use_chat, system_prompt=config.llm.system_prompt, assistant_prompt=config.llm.assistant_prompt)

    train_dataset = NumericDataset(train_data)
    test_dataset = NumericDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=partial(pad_collate_fn, pad_id=pad_id, mask_num=config.occamnet.mask_num), batch_size=config.trainer.train_batch_size, pin_memory=True, drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=partial(pad_collate_fn,pad_id=pad_id,mask_num=config.occamnet.mask_num), batch_size=config.trainer.test_batch_size, pin_memory=True, drop_last=True,
    )
    validation_dataset = json.load(open(os.path.join(config.dirs.data_dir, config.trainer.validation_dataset),"r"))

    # Load model
    if config.llm.debug:
        accelerator.print(f"Creating small LLaMA model for debugging")
        llm_config = LlamaConfig(num_hidden_layers=2, hidden_size=8, intermediate_size=8,  num_attention_heads=4)
        llm = AutoModelForCausalLM.from_config(llm_config)   
        if config.llm.load_in_f16:
            llm.to(dtype=torch.float16)
    else:
        accelerator.print(f"Loading pretrained LLM")
        print(config.llm.load_in_f16)
        try:
            llm = AutoModelForCausalLM.from_pretrained(
                os.path.join(config.dirs.llm_dir,config.llm.release,config.llm.version),
                device_map="auto",
                torch_dtype=torch.float16 if config.llm.load_in_f16 else torch.float32,
            )
        except:
            llm = AutoModelForCausalLM.from_pretrained(
                os.path.join(config.llm.release,config.llm.version),
                device_map="auto",
                torch_dtype=torch.float16 if config.llm.load_in_f16 else torch.float32,
            )

        if config.llm.load_in_f16:
            try:
                llm.to(dtype=torch.float16)
            except:
                pass
            
    model = OccamLLM(llm, tokenizer, config.occamnet)


    if config.llm.freeze:
        print("Freezing LLM")
        for param in model.llm.parameters():
            param.requires_grad = False
        accelerator.print(model)
        accelerator.print(f"LLM params: {sum(p.numel() for p in model.llm.parameters() if p.requires_grad):,}")
        accelerator.print(f"OccamNet Head params: {sum(p.numel() for p in model.occamnet_loss_head.parameters() if p.requires_grad):,}")
    else:
        accelerator.print("Creating LORA adapter")
        # LoRA configuration
        peft_config = LoraConfig(
            inference_mode=False, r=config.llm.lora.r, lora_alpha=config.llm.lora.alpha, lora_dropout=config.llm.lora.dropout,
            target_modules=config.llm.lora.target_modules, modules_to_save=config.llm.lora.modules_to_save
        )
        model.create_lora_adapter(peft_config) 
        accelerator.print(model)
        accelerator.print(f"OccamNet Head params: {sum(p.numel() for p in model.occamnet_loss_head.parameters() if p.requires_grad):,}")
        accelerator.print("LLM: ")
        if accelerator.is_main_process:
            model.llm.print_trainable_parameters()
            
    accelerator.print("Trainable parameters:")
    for pn, p in model.named_parameters():
        if p.requires_grad:
            accelerator.print(pn, p.data.size(), p.device)




    # Setup optimizer and scheduler
    accelerator.print("Setting up optimizer and scheduler")  
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)

    if config.optimizer.scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.warmup_pct*config.trainer.num_epochs*len(train_dataloader),
            num_training_steps=config.trainer.num_epochs*len(train_dataloader),
        )
    elif config.optimizer.scheduler == "cosine":
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=int(config.trainer.num_epochs*len(train_dataloader)/config.optimizer.gradient_accumulation_steps),
            max_lr=config.optimizer.lr,
            pct_start=config.optimizer.warmup_pct,
            div_factor=25
        )
    elif config.optimizer.scheduler == "step":
        lr_scheduler = StepLR(
            optimizer, 
            step_size=1, 
            gamma=config.optimizer.gamma)
    else:
        raise Exception(f"Scheduler '{config.optimizer.scheduler}' not implemented")


    # Prepare model for distributed training
    accelerator.print("Preparing model for distributed training")
    model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
       model, train_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    accelerator.print("Untrained parameters:")
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            accelerator.print(pn, p.data.size(), p.device)

    # Train
    accelerator.print("Start finetuning")
    global_step = 1

    # Train metrics
    train_loss = []
    train_id_loss = []
    train_id_examples = []
    train_num_loss = []
    train_num_examples = []
    train_texts = []
    train_results = []
    train_pred_ops = []
    train_pred_nums = []

    for epoch in range(1, config.trainer.num_epochs+1):

        accelerator.print(f"Epoch {epoch}")
        model.train()

        for step, (model_inputs, texts, results) in enumerate(tqdm(train_dataloader)):
            # Perform gradient accumulation
            if global_step % config.optimizer.gradient_accumulation_steps == 0:
                outputs: OccamLLMOutput = model(**model_inputs)
                id_loss = outputs.id_loss
                id_examples = outputs.id_examples
                num_loss = outputs.num_loss
                num_examples = outputs.num_examples
                loss = num_loss / num_examples
                accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)
                optimizer.step()
                if config.optimizer.scheduler in ["linear","cosine"]:
                    lr_scheduler.step()
                optimizer.zero_grad()
            else:
                with accelerator.no_sync(model):
                    outputs = model(**model_inputs)
                    id_loss = outputs.id_loss
                    id_examples = outputs.id_examples
                    num_loss = outputs.num_loss
                    num_examples = outputs.num_examples
                    loss = num_loss / num_examples
                    accelerator.backward(loss / config.optimizer.gradient_accumulation_steps)


            # Loss
            train_loss.append(accelerator.gather(loss).mean().detach().item())
            train_id_loss.append(accelerator.gather(id_loss).sum().detach().item())
            train_num_loss.append(accelerator.gather(num_loss).sum().detach().item())
            train_num_examples.append(accelerator.gather(num_examples).sum().detach().item())
            train_id_examples.append(accelerator.gather(id_examples).sum().detach().item())

            # Preds
            pred_ops = [[row[j] for j in range(len(row)) if abs(model_inputs["num_labels"][i,j].item() - config.occamnet.mask_num) > 1.e-5] for i, row in enumerate(outputs.occamnet_ops)]
            pred_nums = [[row[j].item() for j in range(len(row)) if abs(model_inputs["num_labels"][i,j].item() - config.occamnet.mask_num) > 1.e-5] for i, row in enumerate(outputs.pred_nums)]
            train_pred_ops += pred_ops
            train_pred_nums += pred_nums
            train_texts += texts
            train_results += results

            # Log to tensorboard
            if accelerator.is_main_process:
                writer.add_scalar("Loss/train_iter",train_loss[-1], global_step)
                writer.add_scalar("IdLoss/train_iter",train_id_loss[-1]/train_id_examples[-1], global_step)
                writer.add_scalar("NumLoss/train_iter", train_num_loss[-1]/train_num_examples[-1], global_step)

                
            # Evaluation condition
            if global_step % config.trainer.eval_every == 0:
                
                # Test metrics
                test_loss = []
                test_id_loss = []
                test_id_examples = []
                test_num_loss = []
                test_num_examples = []
                test_pred_ops = []
                test_pred_nums = []
                test_texts = []
                test_results = []

                accelerator.print(f"Evaluation at step {global_step}")
                model.eval()

                for test_step, (model_inputs, texts, results) in enumerate(tqdm(test_dataloader)):
                    
                    # Forward pass
                    with torch.no_grad() as A, accelerator.no_sync(model) as B:
                        outputs = model(**model_inputs)
                        id_loss = outputs.id_loss
                        id_examples = outputs.id_examples
                        num_loss = outputs.num_loss
                        num_examples = outputs.num_examples
                        loss = num_loss / num_examples
                    
                    # Loss
                    test_loss.append(accelerator.gather(loss).mean().detach().item())
                    test_id_loss.append(accelerator.gather(id_loss).sum().detach().item())
                    test_num_loss.append(accelerator.gather(num_loss).sum().detach().item())
                    test_num_examples.append(accelerator.gather(num_examples).sum().detach().item())
                    test_id_examples.append(accelerator.gather(id_examples).sum().detach().item())

                    # Preds
                    pred_ops = [[row[j] for j in range(len(row)) if abs(model_inputs["num_labels"][i,j].item() - config.occamnet.mask_num) > 1.e-5] for i, row in enumerate(outputs.occamnet_ops)]
                    pred_nums = [[row[j].item() for j in range(len(row)) if abs(model_inputs["num_labels"][i,j].item() - config.occamnet.mask_num) > 1.e-5] for i, row in enumerate(outputs.pred_nums)]
                    test_pred_ops += pred_ops
                    test_pred_nums += pred_nums
                    test_texts += texts
                    test_results += results

                # Eval metrics
                train_epoch_loss = sum(train_loss) / len(train_loss)
                train_epoch_id_loss = sum(train_id_loss) / sum(train_id_examples)
                train_epoch_num_loss = sum(train_num_loss) / sum(train_num_examples)
                test_epoch_loss = sum(test_loss) / len(test_loss)
                test_epoch_id_loss = sum(test_id_loss) / sum(test_id_examples)
                test_epoch_num_loss = sum(test_num_loss) / sum(test_num_examples)

                
                # Training examples
                accelerator.print("TRAIN")
                for i in range(5):
                    accelerator.print("Sentence: " + train_texts[i] + "\nResult: " + "{:.2f}".format(train_results[i]) + "\nPredictions:\n" + '\n'.join([f'{n}  :  ' + op for n, op in zip(train_pred_nums[i],train_pred_ops[i])]))
                accelerator.print("TESTS")
                for i in range(5):
                    accelerator.print("Sentence: " + test_texts[i] + "\nResult: " + "{:.2f}".format(test_results[i]) + "\nPrediction:\n" + '\n'.join([f'{n}  :  ' + op  for n, op in zip(test_pred_nums[i],test_pred_ops[i])]))
                
                # Validation
                accelerator.print("VALIDATION")
                prev_T = model.occamnet_loss_head.occamnet.t_softmax_layer.temperature
                model.occamnet_loss_head.occamnet.t_softmax_layer.temperature = 0.0001
                val_results = eval_occam_llm(validation_dataset, model, tokenizer, pad_id=pad_id, num_inputs=config.occamnet.num_inputs, mask_num=config.occamnet.mask_num, append_bos=config.llm.append_bos, use_chat=use_chat, system_prompt=config.llm.system_prompt, assistant_prompt=config.llm.assistant_prompt, do_print=True)
                val_epoch_acc = 100 * sum(row["is_correct"] for row in val_results) / len(val_results)
                model.occamnet_loss_head.occamnet.t_softmax_layer.temperature = prev_T

                # Summary
                accelerator.print(f"{savestring=} {global_step=}:" + "\n" + \
                                  f"{train_epoch_loss=}\n{train_epoch_id_loss=}\n{train_epoch_num_loss=}\n" + \
                                  f"{test_epoch_loss=}\n{test_epoch_id_loss=}\n{test_epoch_num_loss=}\n" + \
                                  f"{val_epoch_acc=} %")  

                # Log to tensorboard
                if accelerator.is_main_process:
                    # Log to tensorboard
                    writer.add_scalar("Loss/train",train_epoch_loss,global_step)
                    writer.add_scalar("IdLoss/train",train_epoch_id_loss,global_step)
                    writer.add_scalar("NumLoss/train",train_epoch_num_loss,global_step)
                    writer.add_scalar("Loss/test",test_epoch_loss,global_step)
                    writer.add_scalar("IdLoss/test",test_epoch_id_loss,global_step)
                    writer.add_scalar("NumLoss/test",test_epoch_num_loss,global_step)
                    writer.add_scalar("Accuracy/val",val_epoch_acc,global_step)

                # Reset train metrics
                train_loss = []
                train_id_loss = []
                train_id_examples = []
                train_num_loss = []
                train_num_examples = []
                train_pred_ops = []
                train_pred_nums = []
                train_texts = []
                train_results = []

                # End evaluation
                model.train()     

            # Save checkpoints
            if global_step % config.trainer.save_every == 0:
                save_to_path = os.path.join(checkpoint_dir,f"STEP{global_step}")
                if not os.path.exists(save_to_path) and accelerator.is_main_process:
                    os.makedirs(save_to_path)

                accelerator.print(f"Saving checkpoint at step {global_step} to {save_to_path}")
                model.save_checkpoint(save_to_path)
                if accelerator.is_main_process:
                    torch.save(dict(config), os.path.join(save_to_path,"config.pth"))
            
            global_step += 1
                
        if config.optimizer.scheduler in ["step"]:
            lr_scheduler.step()

        
    writer.flush()
    writer.close()

    accelerator.print("Training done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type = str, help="File (.yaml) with configuration for finetuning", default="none")
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs)
    args = parser.parse_args()

    main(args)