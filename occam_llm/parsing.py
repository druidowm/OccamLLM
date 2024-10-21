import re
import math

def get_dec_places(num):
    SPLIT_PATTERN = r'[.eE]'
    parts = re.split(SPLIT_PATTERN, num)
    if len(parts) == 1:
        return 0
    elif len(parts) == 2:
        return len(parts[1])
    elif len(parts) == 3:
        return max(len(parts[1]) - int(parts[2]),0)


def parse_numbers(text, as_float=True):
    NUM_PATTERN = r'(?<!\w)(-?\d{1,3}(,\d{3})*(\.\d+)?([eE][-+]?\d+)?|-?\d+(\.\d+)?([eE][-+]?\d+)?|(?<!\\)(\bpi\b|\be\b|\u03C0|\317\200))(?!\w)'

    numbers = []

    for catch in re.finditer(NUM_PATTERN, text):
        num = str(catch[0])
        if num == "\u03C0" or num == "pi" or num == "\317\200":
            num = str(math.pi)
        elif num == "e":
            num = str(math.e)
        if "," in num:
            num = num.replace(",","")
        if as_float:
            num = float(num)

        numbers.append(num)
    return numbers