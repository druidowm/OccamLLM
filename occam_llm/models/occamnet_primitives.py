import torch
import torch.nn as nn

# Base class for OccamNet primitives
class OccamNetPrimitive(nn.Module):
    arity: int

    def __init__(self):
        super().__init__()

    # compute the primitive on inputs at position `index`
    def forward(
            self, 
            inputs: torch.Tensor, 
            index: int,
        ) -> torch.Tensor:
        pass

    # get string representation of the primitive applied to inputs
    def apply_string(self, inputs, index):
        return f"{self.__class__.__name__}({', '.join([str(inputs[..., index + i]) for i in range(self.arity)])})"


class Identity(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return inputs[...,index]
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]})"
    

class Addition(OccamNetPrimitive):
    arity: int = 2

    def forward(self, inputs, index):
        return inputs[...,index] + inputs[..., index + 1]
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} + {inputs[index + 1]})"

class Subtraction(OccamNetPrimitive):
    arity: int = 2

    def forward(self, inputs, index):
        return inputs[...,index] - inputs[..., index + 1]
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} - {inputs[index + 1]})"
    
class Product(OccamNetPrimitive):
    arity: int = 2

    def forward(self, inputs, index):
        return inputs[...,index] * inputs[..., index + 1]
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} * {inputs[index + 1]})"

class Division(OccamNetPrimitive):
    arity: int = 2

    def forward(self, inputs, index):
        unsafe_inputs = inputs[..., index + 1]
        safe_inputs = torch.where(torch.isclose(unsafe_inputs, torch.zeros_like(unsafe_inputs)), torch.ones_like(unsafe_inputs), unsafe_inputs)
        return inputs[..., index] / safe_inputs
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} / {inputs[index + 1]})"

  
class Square(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.square(inputs[...,index])
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} ** 2)"

class Cube(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.pow(inputs[...,index], torch.tensor(3))
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} ** 3)"


class Cos(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.cos(inputs[...,index])
    
    def apply_string(self, inputs, index):
        return f"cos({inputs[index]})"

class Sin(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.sin(inputs[..., index])
    
    def apply_string(self, inputs, index):
        return f"sin({inputs[index]})"
    

class Exp(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.exp(torch.min(inputs[...,index], torch.tensor(46)))
    
    def apply_string(self, inputs, index):
        return f"exp({inputs[index]})"

class Log(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.log(torch.max(inputs[...,index], torch.tensor(1.e-20)))
    
    def apply_string(self, inputs, index):
        return f"log({inputs[index]})"


class Root(OccamNetPrimitive):
    arity: int = 1

    def forward(self, inputs, index):
        return torch.sqrt(torch.max(inputs[...,index], torch.tensor(1.e-20)))
    
    def apply_string(self, inputs, index):
        return f"sqrt({inputs[index]})"

class Power(OccamNetPrimitive):
    arity: int = 2

    def forward(self, inputs, index):
        safe_bases = torch.max(inputs[...,index], torch.tensor(1.e-20))
        bound = torch.log(torch.tensor(1.e20))/torch.log(safe_bases)

        safe_powers = torch.where(
            bound >= 0,
            torch.min(inputs[...,index + 1], bound),
            torch.max(inputs[...,index + 1], bound),
        )
        
        return torch.pow(safe_bases, safe_powers)
    
    def apply_string(self, inputs, index):
        return f"({inputs[index]} ^ {inputs[index + 1]})"
