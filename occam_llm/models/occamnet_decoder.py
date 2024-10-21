from typing import List

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from ..config import DictConfig

from .mlp import MLP


class OccamNetLayerDecoder(nn.Module):

    def __init__(
            self, 
            layer_input_size:   int, 
            layer_output_size:  int, 
            llm_config:         PretrainedConfig, 
            occamnet_config:    DictConfig
        ):

        super().__init__()

        self.num_llm_layers = occamnet_config.num_llm_layers # number of layers to average over
        output_size = layer_input_size * layer_output_size   # number of weights to predict for this OccamNet layer

        # MLP to decode OccamNet layer weights from LLM layers
        if occamnet_config.encoding_layers == 1:
            self.mlp = nn.Linear(llm_config.hidden_size, output_size)
        elif occamnet_config.encoding_layers > 1:
            self.mlp = MLP(llm_config.hidden_size, output_size, occamnet_config.hidden_size, occamnet_config.encoding_layers, occamnet_config.skip_connections)
        else:
            raise Exception("Must have at least on encoding layer!")
        
        # weights to average LLM layers
        self.layer_averaging_weights = nn.Parameter(
            torch.ones(
                self.num_llm_layers, 
                dtype=torch.float32
            )/self.num_llm_layers,
        )

        # shape of OccamNet layer
        self.output_shape = [layer_input_size, layer_output_size]


    def forward(
            self,
            hidden_states: torch.FloatTensor,   # (batch, n_llm_layers, seq_len, hidden_size)
        ) -> torch.FloatTensor:                 # (batch, seq_len, layer_in, layer_out)

        # average LLM layers and decode to occamnet layer
        averaged_hidden_states = torch.einsum("ijkl,j->ikl", hidden_states[:,-self.num_llm_layers:,:,:], self.layer_averaging_weights) # (batch, seq_len, hidden_siz
        output = self.mlp(averaged_hidden_states)
        
        # reshape into OccamNet layer shape
        return output.reshape(
            (
                output.shape[0], 
                output.shape[1], 
                self.output_shape[0], 
                self.output_shape[1],
            ),
        )   # (batch, seq_len, layer_in, layer_out)




class OccamNetDecoder(nn.Module):


    def __init__(
            self, 
            layer_input_sizes:  List[int], 
            layer_output_sizes: List[int], 
            llm_config:         PretrainedConfig,
            occamnet_config:    DictConfig,
        ):

        super().__init__()

        # decoders for each OccamNet layer
        self.layer_decoders = nn.ModuleList([
            OccamNetLayerDecoder(
                layer_input_size,
                layer_output_size,
                llm_config,
                occamnet_config,
            ) for layer_input_size, layer_output_size in zip(layer_input_sizes, layer_output_sizes)
        ])


    def forward(
            self,
            hidden_states: torch.FloatTensor,   # (batch, n_layers, seq_len, hidden_size)
        ) -> list[torch.FloatTensor]:           # [(batch, seq_len, layer_in, layer_out)]
        
        # decode each OccamNet layer from LLM layers
        return [layer_decoder(hidden_states) for layer_decoder in self.layer_decoders]

