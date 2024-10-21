import os
import yaml
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model, PeftModel

from transformers.activations import ACT2FN
ACT2FN["softsign"] = nn.Softsign
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import DictConfig, update_config
from ..parsing import parse_numbers, get_dec_places

from .occamnet_loss import OccamNetLoss


@dataclass
class OccamLLMOutput():
    logits:         torch.FloatTensor

    # Predicitons
    pred_nums:      Optional[torch.FloatTensor]         = None
    router_decision: Optional[torch.FloatTensor]        = None
    occamnet_weights: Optional[List[torch.FloatTensor]] = None
    occamnet_ops:       Optional[List[List[str]]]           = None

    # Losses
    num_loss:       Optional[torch.FloatTensor]     = None
    num_examples:   Optional[torch.LongTensor]      = None
    
    router_loss:  Optional[torch.FloatTensor]       = None
    router_acc:   Optional[torch.LongTensor]        = None
    router_examples:   Optional[torch.LongTensor]   = None
    
    id_loss:        Optional[torch.FloatTensor]     = None
    id_examples:    Optional[torch.LongTensor]      = None

    # Cache
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]  = None
    past_hidden_states: Optional[torch.FloatTensor]             = None



""" OccamLLM class.
"""
class OccamLLM(nn.Module):

    def __init__(
            self, 
            llm: PreTrainedModel, 
            tokenizer: PreTrainedTokenizer,
            occamnet_config_or_path: Union[DictConfig, str],
        ):

        super().__init__()

        # Assign tokenizer
        self.tokenizer = tokenizer
        self.stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        # Assign llm
        self.llm = llm
        self.llm_config = llm.config

        # Create newly initialized OccamNet Loss Head
        if isinstance(occamnet_config_or_path, DictConfig):
            print(occamnet_config_or_path.from_pt)
            if occamnet_config_or_path.from_pt is not None and occamnet_config_or_path.from_pt != "None":
                occamnet_config_or_path = occamnet_config_or_path.from_pt
                occamnet_config = os.path.join(occamnet_config_or_path, "occamnet_config.yaml")
                self.occamnet_config = update_config(occamnet_config)
            else:
                self.occamnet_config = update_config(occamnet_config_or_path)
        else:
            occamnet_config = os.path.join(occamnet_config_or_path, "occamnet_config.yaml")
            self.occamnet_config = update_config(occamnet_config)

        self.occamnet_loss_head = OccamNetLoss(
            llm_config      = self.llm_config,
            occamnet_config = self.occamnet_config,
        )
        self.occamnet_loss_head.to(torch.float64)
        
        # Load pretrained OccamNet Loss Head weights
        if isinstance(occamnet_config_or_path, str):
            self.occamnet_loss_head.load_state_dict(torch.load(os.path.join(occamnet_config_or_path,"occamnet.bin")), strict=False)

        self.occamnet_loss_head = self.occamnet_loss_head.cuda()

        # Add token loss function
        self.id_loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.id_loss_fn = self.id_loss_fn.cuda()

    """ Compute Cross Entropy Loss for language modeling of token ids
    """ 
    def id_loss(
            self,
            logits:     torch.FloatTensor,      # (batch, seq_len, vocab)
            id_labels:     torch.LongTensor,    # (batch, seq_len)
        ) -> Tuple[torch.FloatTensor, int]:     # ((batch), 1)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = id_labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.llm_config.vocab_size)
        shift_labels = shift_labels.view(-1)

        id_loss = self.id_loss_fn(shift_logits, shift_labels)
        id_examples=(shift_labels != -100).sum()
        
        return id_loss, id_examples


    """ Forward pass of the model
    """
    def forward(
            self,
            input_ids:          torch.LongTensor,                   # (batch, seq_len)
            input_numbers:      torch.FloatTensor,                  # (batch, seq_len, m)
            attention_mask:     torch.LongTensor,                   # (batch, seq_len)
            id_labels:          Optional[torch.LongTensor] = None,  # (batch, seq_len)
            num_labels:         Optional[torch.LongTensor] = None,  # (batch, seq_len)
            include_strings:    bool = True,
            past_key_values:    Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_hidden_states: Optional[torch.FloatTensor] = None,
            use_cache:          bool = True,
        ) -> OccamLLMOutput:

        # Embed tokens of sentence
        inputs_embeds = (self.llm.get_input_embeddings())(input_ids)    # (batch, seq_len, hidden_size)
        
        # Forward language modelling head
        outputs = self.llm(  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Move llm output to GPU 0
        for k in outputs.__dict__:
            if isinstance(outputs.__dict__[k], torch.Tensor):
                outputs.__dict__[k] = outputs.__dict__[k].to("cuda:0")
        
        logits = outputs.logits
        id_labels = id_labels.to("cuda:0") if id_labels is not None else None
        input_numbers = input_numbers.to("cuda:0")
        num_labels = num_labels.to("cuda:0") if num_labels is not None else None
        
        # Compute language modeling loss
        id_loss = None
        id_examples = None
        if id_labels is not None:
            id_loss, id_examples = self.id_loss(logits, id_labels)

        # Forward OccamnetHead head
        hidden_states = torch.stack(outputs.hidden_states, dim=1) # (batch, n_layers, trunc_seq_len, hidden_size)

        # Keep track of past hidden states
        if past_hidden_states is not None:
            hidden_states = torch.cat((past_hidden_states, hidden_states), dim=2)

        # Allow the llm to be loaded in lower precision but match higher precision for OccamNet
        hidden_states = hidden_states.to(self.occamnet_loss_head.initial_weights[0].dtype)
        
        # Change to corresponding precision
        input_numbers = input_numbers.to(self.occamnet_loss_head.initial_weights[0].dtype)
        num_labels = num_labels if num_labels is None else num_labels.to(self.occamnet_loss_head.initial_weights[0].dtype)
        
        # Forward OccamNet
        occamnet_loss, num_examples, pred_nums, occamnet_ops, router_decision, router_loss, router_acc, occamnet_weights  = self.occamnet_loss_head(
            hidden_states,
            input_numbers,
            num_labels,
            include_strings=include_strings,
        )

        # Adjust router loss to be per example
        router_examples = None
        if router_loss is not None:
            router_loss = router_loss[id_labels != -100].sum()
            router_acc = router_acc[id_labels != -100].sum()
            router_examples = (id_labels != -100).sum()

        # Gather outputs
        return OccamLLMOutput(
            logits=logits,

            pred_nums=pred_nums,
            router_decision=router_decision,
            occamnet_weights=occamnet_weights,
            occamnet_ops=occamnet_ops,

            num_loss=occamnet_loss,
            num_examples=num_examples,

            router_loss=router_loss,
            router_acc=router_acc,
            router_examples=router_examples,

            id_loss=id_loss,
            id_examples=id_examples,
            
            past_key_values=outputs.past_key_values if use_cache else None,
            past_hidden_states=hidden_states if use_cache else None,
        )
    

    
    """ One step of generation
    """
    def generation_step(
        self, 
        input_ids: torch.LongTensor, 
        input_numbers: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: bool = True,
    ) -> Dict:

        outputs = self(
            input_ids=input_ids,
            input_numbers=input_numbers,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_hidden_states=past_hidden_states,
            use_cache=use_cache,
        )

        logits = outputs.logits[0, -1, :]
        use_logits = not outputs.router_decision.argmax(-1)[0, -1]
        numeric_output = outputs.pred_nums[0,-1]
        return {
            'logits': logits,
            'numeric_output': numeric_output,
            'pred_op': outputs.occamnet_ops[0][-1],
            'use_logits': use_logits,
            'past_key_values': outputs.past_key_values,
            'past_hidden_states': outputs.past_hidden_states,
            'occamnet_ops': outputs.occamnet_ops,
            'occamnet_weights': outputs.occamnet_weights
        }

    
    
    """ Open ended generation
    """
    def generate_text(
        self, 
        input_text:         str, 
        max_new_tokens:     int,
        assistant_prompt:   Optional[str] = None,
        system_prompt:      Optional[str] = None,
        temperature:        Optional[float] = 1,
        top_p:              Optional[float] = 1,
        use_cache:          bool = False,
    ) -> str:
        prev_num_samples = self.occamnet_loss_head.occamnet_config.num_samples
        self.occamnet_loss_head.occamnet_config.num_samples = 1
        
        # Prepare initial inputs
        new_ids = self.tokenizer.apply_chat_template(
            [   
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if assistant_prompt != "":
            new_ids = torch.cat((new_ids, self.tokenizer(assistant_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]),dim=1)
        new_ids = new_ids.to(self.llm.device)

        prompt_id_len = new_ids.shape[1]

        # Save to decode later
        input_size = new_ids.size(1)     

        # Track where does each token come from
        sources = ["Prompt"] * new_ids.size(1)

        # Track OccamNet state
        occamnet_weights = []
        occamnet_ops = []
    
        # Save when use cache
        past_key_values = None
        past_hidden_states = None
        
        # Initialize
        conversation_ids = torch.empty((1,0), dtype=torch.long, device=self.llm.device)
        attention_mask = torch.empty((1,0), dtype=torch.long, device=self.llm.device)
        input_numbers = torch.empty((1,0,2), dtype=torch.float64, device=self.llm.device)
        while conversation_ids.shape[1] + new_ids.shape[1] - prompt_id_len < max_new_tokens:

            # Extend inputs
            conversation_ids = torch.cat((conversation_ids,new_ids),dim=1)
            all_text = self.tokenizer.batch_decode(conversation_ids, skip_special_tokens=False)[0]

            # Extend numbers
            all_numbers  = ["0"] * self.occamnet_config.num_inputs  + parse_numbers(all_text, as_float=False)
            next_dec_places = max(3,min(8, min([get_dec_places(num) for num in all_numbers[-self.occamnet_config.num_inputs:]])))
            new_numbers = [float(num) for num in all_numbers[-self.occamnet_config.num_inputs:]]
            new_input_numbers = torch.tensor([new_numbers] * new_ids.size(1), dtype=torch.float64).unsqueeze(0).to(input_numbers)
            input_numbers = torch.cat((input_numbers, new_input_numbers), dim=1)
            
            # Extend attention mask
            attention_mask = torch.cat((attention_mask, torch.ones_like(new_ids).to(attention_mask)), dim=1)

            # Finish generation
            if conversation_ids[0,-1] in self.stop_tokens:
                break


            output = self.generation_step(
                        input_ids=new_ids if use_cache else conversation_ids,
                        input_numbers=input_numbers,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        past_key_values=past_key_values,
                        past_hidden_states=past_hidden_states,
                    )
            past_key_values = output['past_key_values']
            past_hidden_states = output['past_hidden_states']
            
            # Follow switch decision
            if output['use_logits']:
                # Just sample next token
                next_token_id = self.sample_from_logits(output['logits'], temperature=temperature, top_p=top_p)
                new_ids = torch.tensor([next_token_id]).to(conversation_ids).unsqueeze(0)
                sources.append("LLM")
            else:
                # Get OccamNet output and tokenize it
                pred_num = output["numeric_output"]
                pred_num_int = output["numeric_output"].to(torch.int64)
                if pred_num_int == pred_num:
                    num_text = "{:d}\n\n".format(pred_num_int)
                else:
                    num_text = f"{pred_num:.{next_dec_places}f}\n\n"
                new_ids = self.tokenizer(num_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(conversation_ids)
                sources += ["OccamNet"] * new_ids.size(1)

            occamnet_weights.append([w.detach().cpu().numpy() for w in output['occamnet_weights']])
            occamnet_ops.append(output['occamnet_ops'])


        # Decode input and response text
        input = self.tokenizer.decode(conversation_ids[0,:input_size], skip_special_tokens=False)
        response = self.tokenizer.batch_decode(conversation_ids[0,input_size:], skip_special_tokens=False)
        
        self.occamnet_loss_head.occamnet_config.num_samples = prev_num_samples
        return response, sources[input_size:], input


    @staticmethod
    def sample_from_logits(logits: torch.Tensor, temperature=1, top_p=1) -> int:
        probs = F.softmax(logits/temperature, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= top_p
        # Shift mask one step to include first element above threshold, so as to always include max prob token
        top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  
        top_p_mask[..., 0] = True 

        filtered_probs = sorted_probs * top_p_mask
        filtered_probs /= torch.sum(filtered_probs, dim=-1, keepdim=True)  # Re-normalize probabilities
        
        token_index = torch.multinomial(filtered_probs, num_samples=1)
        sampled_token = sorted_indices[token_index]
        
        return sampled_token



    ##  ADAPTER METHODS  ## 
    """ Load trained LoRA adapter for the LLM
    """
    def load_lora_adapter(self, adapter_dir, is_trainable=False):
        self.llm = PeftModel.from_pretrained(self.llm, adapter_dir, is_trainable=is_trainable)
    
    """ Create new LoRA adapter for the LLM
    """
    def create_lora_adapter(self, peft_config):
        self.llm = get_peft_model(self.llm, peft_config)

    """ Merge LoRA weigths with original LLM weights
    """
    def merge_lora_adapter(self):
        self.llm = self.llm.merge_and_unload()


    """ Unload LoRA adapter from LLM
    """
    def unload_lora_adapter(self):
        self.llm = self.llm.unload()

    ##  SAVING METHODS  ##

    """ Save trained LoRA adapter
    """
    def save_lora_adapter(self, adapter_dir):
        if getattr(self.llm, "peft_type", None) is None:
            print("No adapter loaded. Nothing saved.")
            return

        if not os.path.exists(adapter_dir):
            os.makedirs(adapter_dir)
        self.llm.save_pretrained(adapter_dir)

    """ Save OccamNet Numeric Head weights
    """
    def save_occamnet(self, occamnet_dir):
        yaml.dump(dict(self.occamnet_config), open(os.path.join(occamnet_dir, "occamnet_config.yaml"),"w"), default_flow_style=False)
        torch.save(self.occamnet_loss_head.state_dict(), os.path.join(occamnet_dir,"occamnet.bin"))
    
    """Save OccamNet Numeric Head and adapter weights
    """
    def save_checkpoint(self, checkpoint_dir):
        self.save_occamnet(checkpoint_dir)
        self.save_lora_adapter(checkpoint_dir)
   
   
