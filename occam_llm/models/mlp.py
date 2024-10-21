import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(
        self, 
        input_size:         int, 
        output_size:        int, 
        hidden_size:        int,
        encoding_layers:    int,
        skip_connections:   bool,
    ):
        super().__init__()

        self.skip_connections = skip_connections

        modules = [nn.Linear(input_size, hidden_size)]
        modules += [nn.Linear(hidden_size, hidden_size) for _ in range(encoding_layers - 2)]
        modules += [nn.Linear(hidden_size, output_size)]

        self.layers = nn.ModuleList(modules)

        self.activation = nn.GELU()


    def forward(self, inputs: torch.FloatTensor):
        inputs = self.activation(self.layers[0](inputs))

        for layer in self.layers[1:-1]:
            if self.skip_connections:
                inputs = self.activation(layer(inputs)) + inputs
            else:
                inputs = self.activation(layer(inputs))

        return self.layers[-1](inputs)

