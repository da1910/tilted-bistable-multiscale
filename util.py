from typing import Dict
import numpy as np
from functools import reduce
from bisect import bisect_left


def create_output_file(source_filename: str, destination_filename: str, replacements: Dict[str, str]) -> None:
    with open(source_filename, 'r', encoding='utf8') as f_in:
        with open(destination_filename, 'w', encoding='utf8') as f_out:
            for line in f_in.readlines():
                line = reduce(lambda a, kv: a.replace(*kv), replacements.items(), line)
                f_out.write(line)


def interp1(x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    output = np.ndarray((len(new_x), y.shape[1]))
    for row_index, x_target in enumerate(new_x):
        output[row_index, :] = interp1_inner(x, y, x_target)
    return output


def interp1_inner(x: np.ndarray, y: np.ndarray, new_x: float) -> np.ndarray:
    index = bisect_left(x, new_x)
    x_interval = x[index] - x[index - 1]
    x_diff = new_x - x[index]
    ratio = x_diff / x_interval
    y_diff = y[index, :] - y[index - 1, :]
    return y[index - 1, :] + ratio * y_diff


def to_fortran_string(number: float) -> str:
    return '{:.3e}'.format(number).replace('e', 'D')