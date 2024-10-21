import json
import os
import random

import argparse

from occam_llm.config import update_config

# use argparse to get "split"
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train", choices=["train","test","val"])
parser.add_argument("--dataset_name", type=str, default="train_multiarith_synthetic")
parser.add_argument("--max_levels", type=int, default=1)
parser.add_argument("--max_entries", type=int, default=20000)
parser.add_argument("--max_result", type=int, default=1000)
parser.add_argument("--integer_range", type=tuple, default=(-100, 100))

args = parser.parse_args()





# dictionary of operator precedence and incidence probability, with an
# evaluator added just for fun.
operators = {
    '*': {'prec': 20, 'prob': .15, 'eval': lambda a, b: a*b},
    '/': {'prec': 20, 'prob': .35, 'eval': lambda a, b: a/b},
    '+': {'prec': 30, 'prob': .15, 'eval': lambda a, b: a+b},
    '-': {'prec': 30, 'prob': .35, 'eval': lambda a, b: a-b}
}

random.seed()

# A node in an expression tree
class expression(object):
    def __init__(self):
        super(expression, self).__init__()

    def precedence(self):
        return -1

    def eval(self):
        return 0

    @classmethod
    def create_random(cls, level):
        if level == 0:
            is_op = True
        elif level == args.max_levels:
            is_op = False
        else:
            is_op = random.random() <= 1.0 - pow(level/args.max_levels, 2.0)

        if is_op:
            return binary_expression.create_random(level)
        else:
            return integer_expression.create_random(level)

class integer_expression(expression):
    def __init__(self, value):
        super(integer_expression, self).__init__()

        self.value = value

    def __str__(self):
        return self.value.__str__()

    def precedence(self):
        return 0

    def eval(self):
        return self.value
    
    def num_ops(self):
        return 0

    @classmethod
    def create_random(cls, level):
        return integer_expression(random.randint(args.integer_range[0],
                                                 args.integer_range[1]))

class binary_expression(expression):
    def __init__(self, symbol, left_expression, right_expression):
        super(binary_expression, self).__init__()

        self.symbol = symbol
        self.left = left_expression
        self.right = right_expression

    def eval(self):
        f = operators[self.symbol]['eval']
        return f(self.left.eval(), self.right.eval())

    @classmethod
    def create_random(cls, level):
        symbol = None

        # Choose an operator based on its probability distribution
        r = random.random()
        cumulative = 0.0
        for k, v in operators.items():
            cumulative += v['prob']
            if r <= cumulative:
                symbol = k
                break

        assert symbol != None

        left = expression.create_random(level + 1)
        right = expression.create_random(level + 1)

        return binary_expression(symbol, left, right)

    def precedence(self):
        return operators[self.symbol]['prec']
    
    def num_ops(self):
        return self.left.num_ops() + self.right.num_ops() + 1

    def __str__(self):
        left_str = self.left.__str__()
        right_str = self.right.__str__()
        op_str = self.symbol

        # Nice to have space around low precedence operators
        if operators[self.symbol]['prec'] >= 30:
            op_str = ' ' + op_str + ' '

        return "(" + left_str + op_str + right_str + ")"
    

def get_expressions_with_condition(num_expressions, condition = None):
    ds = []
    while len(ds) < num_expressions:
        expr = expression.create_random(0)

        print(f"Expression: {expr}")
        print(f"Num Ops: {expr.num_ops()}")

        if condition == None or condition(expr):
            try:
                value = float(expr.eval())
                if abs(value) < args.max_result:
                    print(expr, '=', value)

                    sample = {}
                    sample["input"] = str(expr) + " = "
                    sample["output"] = value
                    ds.append(sample)
            except:
                pass

    return ds

dirs = update_config("configs/sc_dirs.yaml")
data_dir = dirs.data_dir

if args.split == "test":
    for i in range(1,4):
        ds = get_expressions_with_condition(args.max_entries, lambda x: x.num_ops() == i)

        dataset_path = os.path.join(data_dir,f"{args.dataset_name}_{i+1}.json")

        print(f"Writing dataset to {dataset_path}")

        with open(dataset_path,"w") as ofile:
            json.dump(ds, ofile)

else:
    ds = get_expressions_with_condition(args.max_entries)

    if args.split == "train":
        ds = {"train": ds, "test": ds[-1:]}
        print(len(ds["train"]), len(ds["test"]))
    else:
        print(len(ds))

    dataset_path = os.path.join(data_dir,f"{args.dataset_name}.json")

    print(f"Writing dataset to {dataset_path}")

    with open(dataset_path,"w") as ofile:
        json.dump(ds, ofile)