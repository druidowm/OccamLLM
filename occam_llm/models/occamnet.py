from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn

from ..config import DictConfig

from . import occamnet_primitives


class OccamNetActivationLayer(nn.Module):
    
    primitives: List[occamnet_primitives.OccamNetPrimitive]
    arities: torch.LongTensor
    num_inputs: int
    num_outputs: int
    skip_connection: bool

    def __init__(self, primitives: List[str], prev_output_size: int, skip_connection: bool = False):
        super().__init__()

        primitive_dict = {}
        for name, obj in occamnet_primitives.__dict__.items():
            try:
                if issubclass(obj, occamnet_primitives.OccamNetPrimitive) and name != "OccamNetPrimitive":
                    primitive_dict[name] = obj
            except:
                pass

        self.primitives = nn.ModuleList([
            primitive_dict[primitive]() for primitive in primitives
        ])

        self.arities = torch.tensor([primitive.arity for primitive in self.primitives], dtype = torch.long)
        self.num_inputs = sum(self.arities)

        if skip_connection:
            self.num_outputs = len(self.primitives) + prev_output_size
        else:
            self.num_outputs = len(self.primitives)

        self.skip_connection = skip_connection

    def forward(
            self, 
            arguments_layer: torch.FloatTensor,                 # (batch, num_inputs)
            prev_image_layer: Union[torch.FloatTensor,None],    # (batch, num_inputs)
        ) -> torch.FloatTensor:                                 # (batch, num_outputs)

        outputs = []
        index = 0

        for primitive in self.primitives:
            outputs.append(primitive(arguments_layer, index))
            index += primitive.arity

        if self.skip_connection:
            return torch.cat([torch.stack(outputs, dim = -1), prev_image_layer], dim = -1)
        
        return torch.stack(outputs, dim = -1)
    
    def propagate_mask(
            self,
            output_mask: torch.BoolTensor                       # (num_samples, batch, seq_len, num_outputs)
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  # [(num_samples, batch, seq_len num_inputs), (num_samples, batch, seq_len, num_skip)]

        if self.skip_connection:
            skip_mask = output_mask[..., len(self.primitives):]
            output_mask = output_mask[..., :len(self.primitives)]
        else:
            skip_mask = None
        
        input_mask = torch.repeat_interleave(
            input = output_mask,
            repeats = self.arities,
            dim = -1,
        )

        return input_mask, skip_mask

    
    def apply_string(
            self,
            image_layer: List[str],       # (num_inputs)
            prev_image_layer: List[str],  # (prev_num_inputs)
        ) -> List[str]:         # (num_outputs)

        outputs = []
        index = 0

        for primitive in self.primitives:
            outputs.append(primitive.apply_string(image_layer, index))
            index += primitive.arity

        if self.skip_connection:
            return outputs + prev_image_layer

        return outputs
    

class TSoftmaxLayer(nn.Module):
    softmax_weights: bool
    temperature: float

    def __init__(self, softmax_weights: bool, temperature: float):
        super().__init__()

        self.softmax_weights = softmax_weights
        self.temperature = temperature

    def forward_eql(
            self,
            inputs: torch.FloatTensor,          # (batch, seq_len, num_in)
            weights: torch.FloatTensor,         # (batch, seq_len, layer_in, layer_out)
        ) -> torch.FloatTensor:                 # (batch, seq_len, num_out)

        if self.softmax_weights:
            weights = nn.functional.softmax(weights/self.temperature, 2)

        return torch.einsum("...ij,...i->...j", weights, inputs)
    
    def forward_samples(
            self,
            inputs: torch.FloatTensor,          # (samples, batch, seq_len, num_in)
            samples: torch.LongTensor,          # (samples, batch, seq_len, num_out)
        ) -> torch.FloatTensor:                 # (samples, batch, seq_len, num_out)

        return torch.gather(inputs, 3, samples)
    
    def sample_paths(
            self,
            weights: torch.FloatTensor,         # (batch, seq_len, layer_in, layer_out)
            num_samples: int,
        ) -> torch.LongTensor:                  # (num_samples, batch, seq_len, layer_out)

        return torch.distributions.Categorical(
            logits = weights.transpose(2,3)/self.temperature
        ).sample((num_samples,))
    
    def propagate_masks(
            self,
            weights: torch.FloatTensor,                     # (batch, seq_len, layer_in, layer_out)
            paths: torch.LongTensor,                        # (num_samples, batch, seq_len, layer_out)
            masks: torch.BoolTensor,                        # (num_samples, batch, seq_len, layer_out)
            skip_masks: Optional[torch.BoolTensor] = None,  # (num_samples, batch, seq_len, layer_in)
        ) -> torch.BoolTensor:                              # (num_samples, batch, seq_len, layer_in)

        if skip_masks is None:
            new_masks = torch.zeros(
                masks.shape[0],
                masks.shape[1],
                masks.shape[2],
                weights.shape[2],
                dtype = torch.bool,
            )
        else:
            new_masks = skip_masks.clone()

        new_masks[
            torch.arange(paths.shape[0]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, paths.shape[1], paths.shape[2], paths.shape[3]),
            torch.arange(paths.shape[1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(paths.shape[0], -1, paths.shape[2], paths.shape[3]),
            torch.arange(paths.shape[2]).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(paths.shape[0], paths.shape[1], -1, paths.shape[3]),
            paths,
        ] = masks

        return new_masks
    
    def get_log_probs(
            self,
            paths: torch.LongTensor,            # (num_samples, batch, seq_len, layer_out)
            masks: torch.BoolTensor,            # (num_samples, batch, seq_len, layer_out)
            weights: torch.FloatTensor,         # (batch, seq_len, layer_in, layer_out)
        ) -> torch.FloatTensor:                 # (num_samples, batch, seq_len)

        log_probs = torch.log_softmax(weights/self.temperature, 2)

        log_probs = log_probs[
            torch.arange(weights.shape[0])
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(
                    paths.shape[0], 
                    -1, 
                    paths.shape[2], 
                    paths.shape[3]
                ),
            torch.arange(weights.shape[1])
                .unsqueeze(0)
                .unsqueeze(1)
                .unsqueeze(3)
                .expand(
                    paths.shape[0], 
                    paths.shape[1], 
                    -1, 
                    paths.shape[3],
                ),
            paths,
            torch.arange(weights.shape[3])
                .unsqueeze(0)
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(
                    paths.shape[0], 
                    paths.shape[1], 
                    paths.shape[2], 
                    -1,
                ),
        ]                                   # (num_samples, batch, seq_len, layer_out)

        log_probs[torch.logical_not(masks)] = 0

        return torch.sum(log_probs, 3)

    def apply_weights_string(
            self,
            weight: torch.FloatTensor,                      # (layer_in, layer_out)
            inputs: List[str],                              # (layer_in)
            sample: bool = False,
            layer_path: Optional[torch.FloatTensor] = None  # (layer_out)
        ) -> List[str]:                                     # (layer_out)

        if sample:
            if layer_path is None:
                sample = torch.argmax(weight, dim = 0)
            else:
                sample = layer_path

            #print(sample)
            return [inputs[sample[i]] for i in range(sample.shape[0])]

        if self.softmax_weights:
            weight = nn.functional.softmax(weight/self.temperature, 0)

        outputs = ["(" for _ in range(weight.shape[1])]
        
        for i in range(weight.shape[1]):
            tolerance = 0.1
            while torch.all(torch.abs(weight[:, i]) < tolerance):
                tolerance /= 10

            for j in range(weight.shape[0]):
                if not -tolerance < weight[j, i] < tolerance:
                    outputs[i] += f"{weight[j, i]:.2f} * {inputs[j]} + "

            outputs[i] = outputs[i][:-3] + ")"

        return outputs



class OccamNet(nn.Module):
    activation_layers: List[OccamNetActivationLayer]
    t_softmax_layer: TSoftmaxLayer

    num_inputs: int
    
    def __init__(self, occamnet_config: DictConfig):
        super().__init__()

        prev_layer_size = occamnet_config.num_inputs
        activation_layers = []

        for i in range(occamnet_config.num_layers):
            activation_layers.append(
                OccamNetActivationLayer(
                    occamnet_config.primitives * (2 ** (occamnet_config.num_layers - i - 1)), 
                    prev_layer_size,
                    skip_connection=occamnet_config.occamnet_skip_connections
                )
            )
            prev_layer_size = activation_layers[-1].num_outputs

        self.activation_layers = nn.ModuleList(activation_layers)
        self.t_softmax_layer = TSoftmaxLayer(occamnet_config.softmax_weights, occamnet_config.temperature)

        self.num_inputs = occamnet_config.num_inputs

    def get_weight_shapes(self) -> List[tuple]:
        weight_inputs = [self.num_inputs] + [layer.num_outputs for layer in self.activation_layers]
        weight_outputs = [layer.num_inputs for layer in self.activation_layers] + [1]

        return [[weight_inputs[i], weight_outputs[i]] for i in range(len(weight_inputs))]
    
    def get_equalized_weights(self) -> List[torch.FloatTensor]:
        weight_shapes = self.get_weight_shapes()


        weights = [torch.zeros(weight_shapes[0], dtype = torch.float32, requires_grad=False, device="cuda")]

        prev_logprobs = torch.zeros((self.num_inputs,), dtype = torch.float32, device="cuda")
        current_logprob = torch.log_softmax(weights[0][:,0]/self.t_softmax_layer.temperature, dim = 0)[0]

        for i in range(1,len(weight_shapes)):
            logprobs = []
            for arity in self.activation_layers[i-1].arities:
                logprobs.append(arity*current_logprob)

            logprobs = torch.tensor(logprobs, dtype = torch.float32, device="cuda")
            if self.activation_layers[i-1].skip_connection:
                logprobs = torch.concatenate((logprobs,prev_logprobs), axis=0)
            
            prev_logprobs = logprobs

            weight_inits = self.t_softmax_layer.temperature * (torch.min(logprobs)-logprobs)
            
            expanded_weights = weight_inits.unsqueeze(1).expand(-1, weight_shapes[i][1])
            expanded_weights.requires_grad = False
            weights.append(expanded_weights)
            
            current_logprob = logprobs[0] + torch.log_softmax(weights[i][:,0]/self.t_softmax_layer.temperature, dim = 0)[0]
        
        return weights

    def forward_eql(
            self,
            image_layer: torch.FloatTensor,     # (batch, seq_len, num_in)
            weights: List[torch.FloatTensor],   # [(batch, seq_len, layer_in, layer_out)]
        ) -> torch.FloatTensor:                 # (batch, seq_len)

        for weight, layer in zip(weights[:-1], self.activation_layers):
            arguments_layer = self.t_softmax_layer.forward_eql(image_layer, weight)

            image_layer = layer(arguments_layer, image_layer)

        outputs = self.t_softmax_layer.forward_eql(image_layer, weights[-1])

        return outputs[...,0]
    
    def get_masks(
            self,
            weights: List[torch.FloatTensor],       # [(batch, seq_len, layer_in, layer_out)]
            samples: List[torch.LongTensor]     # [(num_samples, batch, seq_len, layer_out)]
        ) -> List[torch.BoolTensor]:            # [(num_samples, batch, seq_len, layer_out)]

        input_masks = [
            torch.ones(
                (samples[0].shape[0], samples[0].shape[1], samples[0].shape[2], 1),
                dtype = torch.bool,
            )
        ]

        skip_mask = None

        for weights_layer, sample_layer, activation_layer in zip(weights[-1:0:-1], samples[-1:0:-1], self.activation_layers[::-1]):
            output_mask = self.t_softmax_layer.propagate_masks(
                weights=weights_layer,
                paths=sample_layer,
                masks=input_masks[0],
                skip_masks=skip_mask,
            )

            input_mask, skip_mask = activation_layer.propagate_mask(output_mask)

            input_masks.insert(0, input_mask)

        return input_masks
    
    def get_log_probs(
            self,
            paths: List[torch.LongTensor],          # [(num_samples, batch, seq_len, layer_out)]
            weights: List[torch.FloatTensor],       # [(batch, seq_len, layer_in, layer_out)]
        ) -> torch.FloatTensor:                     # (num_samples, batch, seq_len)

        masks = self.get_masks(weights, paths)

        log_probs = torch.zeros_like(paths[0][:,:,:,0], dtype=weights[0].dtype)

        for path, mask, weight in zip(paths, masks, weights):
            log_probs += self.t_softmax_layer.get_log_probs(path, mask, weight)

        return log_probs
    
    def sample_functions_and_probs(
            self,
            weights:    List[torch.FloatTensor],                            # [(batch, seq_len, layer_in, layer_out)]
            num_samples: int,
        ) -> tuple[List[torch.LongTensor], torch.FloatTensor]:              # ([(num_samples, batch, seq_len, layer_out)], (num_samples, batch, seq_len))
        
        samples = [self.t_softmax_layer.sample_paths(layer_weights, num_samples) for layer_weights in weights]

        #masks = self.get_masks(samples)
        log_probs = self.get_log_probs(samples, weights)

        return samples, log_probs
    
    def sample_argmax(
            self,
            weights: List[torch.FloatTensor],   # [(batch, seq_len, layer_in, layer_out)]
        ) -> List[torch.LongTensor]:           # [(batch, seq_len, layer_out)]

        samples, logprobs = self.sample_functions_and_probs(weights, 100)

        argmax = torch.argmax(logprobs, dim = 0, keepdim=True).unsqueeze(-1)

        samples = [torch.gather(sample, 0, argmax.expand(-1, -1, -1, sample.shape[3]))[0] for sample in samples]

        #samples = [torch.argmax(weight, dim = 2) for weight in weights]

        return samples
    
    @torch.no_grad
    def forward_samples(
            self,
            image_layer: torch.FloatTensor,      # (batch, seq_len, num_in)
            samples: List[torch.LongTensor],     # [(num_samples, batch, seq_len, layer_out)]
        ) -> torch.FloatTensor:                  # (batch, seq_len)

        image_layer = image_layer.unsqueeze(0).expand(samples[0].shape[0], -1, -1, -1)
        
        for sample_layer, activation_layer in zip(samples[:-1], self.activation_layers):
            arguments_layer = self.t_softmax_layer.forward_samples(image_layer, sample_layer)

            image_layer = activation_layer(arguments_layer, image_layer)

        outputs = self.t_softmax_layer.forward_samples(image_layer, samples[-1])

        return outputs[...,0]
    
    def to_string(
            self,
            weights:    List[torch.FloatTensor],            # [(batch, seq_len, layer_in, layer_out)]
            inputs:     Optional[torch.FloatTensor] = None, # (batch, seq_len, num_in)
            sample:     bool = False,
            path:       Optional[torch.LongTensor] = None,  # (batch, seq_len, num_out)
        ) -> List[List[str]]:                               # (batch, seq_len)

        outputs = [[None for _ in range(weights[0].shape[1])] for _ in range(weights[0].shape[0])]

        for i in range(weights[0].shape[0]):
            for j in range(weights[0].shape[1]):
                outputs[i][j] = self.to_string_single_example(
                    [weight[i, j] for weight in weights], 
                    inputs[i, j] if inputs is not None else None,
                    sample = sample,
                    path = [layer_path[i,j] for layer_path in path] if path is not None else None,
                )
        
        return outputs

    def to_string_single_example(
            self,
            weights:    List[torch.FloatTensor],            # [(layer_in, layer_out)]
            inputs:     Optional[torch.FloatTensor] = None, # (num_in)
            sample:     bool = False,
            path:       Optional[torch.LongTensor] = None,  # (num_out)
        ) -> str:
        
        if inputs is None:
            strings = [f"x_{i}" for i in range(weights[0].shape[0])]
        else:
            strings = [f"{inputs[i]:.2f}" for i in range(inputs.shape[0])]

        if path is None:
            path = [None] * len(weights)

        for weight, layer, layer_path in zip(weights[:-1], self.activation_layers, path):
            argument_strings = self.t_softmax_layer.apply_weights_string(
                weight, 
                strings, 
                sample = sample,
                layer_path = layer_path
            )

            strings = layer.apply_string(argument_strings, strings)

        strings = self.t_softmax_layer.apply_weights_string(
            weights[-1], 
            strings, 
            sample = sample,
            layer_path = path[-1],
        )

        return strings[0]
