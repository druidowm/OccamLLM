import os
from typing import List, Tuple

import torch
from torch import nn

from transformers import PretrainedConfig

from ..config import DictConfig

from .occamnet import OccamNet
from .mlp import MLP
from .occamnet_decoder import OccamNetDecoder


class OccamNetLoss(nn.Module):
    """
    INCLUDES ALL VARIANTS OF THE OCCAMNET/EQL LOSSES USED
    """
    occamnet_config: DictConfig
    occamnet_decoder: OccamNetDecoder
    occamnet: OccamNet
    mse_loss: nn.MSELoss

    initial_weights: List[torch.FloatTensor]

    def __init__(
        self, 
        llm_config:               PretrainedConfig, 
        occamnet_config:          DictConfig,
    ):

        super().__init__()

        self.occamnet_config = occamnet_config
        self.occamnet = OccamNet(occamnet_config)

        layer_input_sizes = [occamnet_config.num_inputs] + [layer.num_outputs for layer in self.occamnet.activation_layers]
        layer_output_sizes = [layer.num_inputs for layer in self.occamnet.activation_layers] + [1]

        # Decoder from LLM layers to OccamNet weights
        self.occamnet_decoder = OccamNetDecoder(
            layer_input_sizes=layer_input_sizes,
            layer_output_sizes=layer_output_sizes,
            llm_config=llm_config,
            occamnet_config=occamnet_config,
        )
        # Initialize OccamNet weights to uniform distribution
        self.initial_weights = [weight.to(torch.float64) for weight in self.occamnet.get_equalized_weights()]

        
        # Decoder from LLM layers to router decision
        self.num_llm_layers = occamnet_config.num_llm_layers # number of layers to average over
        self.layer_averaging_weights = nn.Parameter(
            torch.ones(
                self.num_llm_layers, 
                dtype=torch.float32
            ) / self.num_llm_layers,
        )
        
         # MLP to decode router decision from LLM layers
        if occamnet_config.encoding_layers == 1:
            self.router = nn.Linear(llm_config.hidden_size, 2)
        elif occamnet_config.encoding_layers > 1:
            self.router = MLP(llm_config.hidden_size, 2, occamnet_config.hidden_size, occamnet_config.encoding_layers, occamnet_config.skip_connections)
        else:
            raise Exception("Must have at least one encoding layer!")
        
        # Compute MSE loss for training OccamNet decoder and XEnt loss for training router decoder
        self.mask_num = occamnet_config.mask_num
        self.mse_loss = nn.MSELoss(reduction="none")
        self.xent_loss = nn.CrossEntropyLoss(reduction="none")


    """ Compute MSE for prediction of numeric values """
    def eql_mse_loss(
            self,
            pred_nums:     torch.FloatTensor,   # (batch, seq_len)
            num_labels:    torch.FloatTensor,   # (batch, seq_len)
        ) -> Tuple[torch.FloatTensor, int]:     # ((batch), 1)
        
        num_loss = self.mse_loss(pred_nums, num_labels)
        num_loss = num_loss[num_labels != self.mask_num].sum()
        num_examples = (num_labels != self.mask_num).sum()
        return num_loss, num_examples

    """ Compute sample-based smoothed cross entropy (the OccamNet categorical loss) """
    def occamnet_categorical_loss(
            self,
            pred_nums:      torch.FloatTensor,      # (num_samples, batch, seq_len)
            pred_logprobs:  torch.FloatTensor,      # (num_samples, batch, seq_len)
            true_nums:      torch.FloatTensor,      # (batch, seq_len)
        ) -> Tuple[torch.FloatTensor, int]:         # ((batch), 1)
        # Compute loss only on labeled numbers
        expand_true_nums = true_nums.unsqueeze(0).expand_as(pred_nums)
        mask = (expand_true_nums != self.mask_num)
        unmask_true_nums = expand_true_nums[mask]
        unmask_pred_nums = pred_nums[mask]
        unmask_pred_logprobs = pred_logprobs[mask]

        mask2 = torch.isclose(unmask_pred_nums, unmask_true_nums)

        if torch.sum(mask2) > 0:
            return -unmask_pred_logprobs[mask2].mean(), mask2.sum()
        
        return -0 * unmask_pred_logprobs.mean(), (mask2.sum() + 1)

    """ Compute sample-based reinforce loss (the OccamNet reinforce loss) """
    def occamnet_RL_loss(
            self,
            pred_nums:      torch.FloatTensor,      # (num_samples, batch, seq_len)
            pred_logprobs:  torch.FloatTensor,      # (num_samples, batch, seq_len)
            true_nums:      torch.FloatTensor,      # (batch, seq_len)
        ) -> Tuple[torch.FloatTensor, int]:         # ((batch), 1)

        # Compute loss only on labeled numbers
        expand_true_nums = true_nums.unsqueeze(0).expand_as(pred_nums)
        mask = (expand_true_nums != self.mask_num)
        unmask_true_nums = expand_true_nums[mask]
        unmask_pred_nums = pred_nums[mask]
        unmask_pred_logprobs = pred_logprobs[mask]

        # Compute loss
        reward = 1 / (torch.abs(unmask_pred_nums - unmask_true_nums) + 0.1)
        return - (reward * unmask_pred_logprobs).sum() / reward.sum(), mask.sum()
    
    def forward_eql(self,
            weights:        List[torch.FloatTensor],    # [(batch, seq_len, layer_in, layer_out)]
            input_numbers:  torch.FloatTensor,          # (batch_size, seq_len, m)
            num_labels:     torch.FloatTensor,          # (batch, seq_len)
            include_strings: bool = False,
        ) -> Tuple[torch.FloatTensor, int, torch.FloatTensor, List[str]]:

        pred_nums = self.occamnet.forward_eql(input_numbers, weights)  # (batch, seq_len)

        if self.occamnet_config.loss_type == "MSE":
            occamnet_loss, num_examples = self.eql_mse_loss(pred_nums, num_labels)
        else:
            raise NotImplementedError()

        string_representation = self.occamnet.to_string(weights, input_numbers) if include_strings else None

        return occamnet_loss, num_examples, pred_nums, string_representation

    def forward_occamnet(self,
            weights: List[torch.FloatTensor],           # [(batch, seq_len, layer_in, layer_out)]
            input_numbers: torch.FloatTensor,           # (batch_size, seq_len, m)
            num_labels: torch.FloatTensor,              # (batch, seq_len)
            include_strings: bool = False,
        ) -> Tuple[torch.FloatTensor, int, torch.FloatTensor, List[str]]:

        weights = [weight + initial_weight for weight, initial_weight in zip(weights, self.initial_weights)]

        samples, logprobs = self.occamnet.sample_functions_and_probs(weights, self.occamnet_config.num_samples)
        pred_nums = self.occamnet.forward_samples(input_numbers, samples)

        num_examples = None
        occamnet_loss = None
        if num_labels is not None:
            if self.occamnet_config.loss_type == "Categorical":
                occamnet_loss, num_examples = self.occamnet_categorical_loss(
                    pred_nums,
                    logprobs,
                    num_labels,
                )
            elif self.occamnet_config.loss_type == "RL":
                occamnet_loss, num_examples = self.occamnet_RL_loss(
                    pred_nums,
                    logprobs,
                    num_labels,
                )

        # Compute prediction using the function with maximum probabilities
        argmax_path = self.occamnet.sample_argmax(weights)
        argmax_output = self.occamnet.forward_samples(input_numbers, [layer.unsqueeze(0) for layer in argmax_path])
        argmax_output = argmax_output.squeeze(0)
        
        if include_strings:
            string_representation = self.occamnet.to_string(weights, input_numbers, sample=True, path=argmax_path)
        else:
            string_representation = None

        return occamnet_loss, num_examples, argmax_output, string_representation

    def forward(
            self,
            hidden_states: torch.FloatTensor,   # (batch_size, n_layers, seq_len, hidden_size)
            input_numbers: torch.FloatTensor,   # (batch_size, seq_len, m)
            num_labels: torch.FloatTensor,      # (batch, seq_len)
            include_strings: bool = False,
        ) -> Tuple[torch.FloatTensor, int, torch.FloatTensor, List[str], torch.FloatTensor, torch.FloatTensor, torch.BoolTensor, List[torch.FloatTensor]]:

        # OccamNet
        weights = self.occamnet_decoder(hidden_states)  # [(batch, seq_len, layer_in, layer_out)]

        # Router
        # average LLM layers and decode to router decision
        averaged_hidden_states = torch.einsum(
            "ijkl,j->ikl",
            hidden_states[:, -self.num_llm_layers:, :, :],
            self.layer_averaging_weights
        )  # (batch, seq_len, hidden_size)
        router_decision = self.router(averaged_hidden_states)  # (batch, seq_len, 2)

        # Router loss
        router_loss = None
        router_acc = None
        if num_labels is not None:
            decision_labels = (num_labels != self.mask_num).to(torch.int64)
            router_loss = self.xent_loss(
                router_decision.view(-1, 2),
                decision_labels.view(-1)
            ).view(num_labels.size(0), -1)
            router_acc = (router_decision.argmax(-1) == decision_labels)
            
        # OccamNet loss
        if self.occamnet_config.loss == "EQL":
            return self.forward_eql(weights, input_numbers, num_labels, include_strings) + (router_decision, router_loss, router_acc, weights)
        elif self.occamnet_config.loss == "OccamNet":
            return self.forward_occamnet(weights, input_numbers, num_labels, include_strings) + (router_decision, router_loss, router_acc, weights)
