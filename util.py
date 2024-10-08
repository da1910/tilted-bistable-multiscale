from typing import Dict
import numpy as np
from functools import reduce


def create_output_file(
    source_filename: str, destination_filename: str, replacements: Dict[str, str]
) -> None:
    with open(source_filename, "r", encoding="utf8") as f_in:
        with open(destination_filename, "w", encoding="utf8") as f_out:
            for line in f_in.readlines():
                line = reduce(lambda a, kv: a.replace(*kv), replacements.items(), line)
                f_out.write(line)


def interp1(x: np.ndarray, y: np.ndarray, new_x: np.ndarray) -> np.ndarray:
    output = np.ndarray((len(new_x), y.shape[1]))
    for row_index, x_target in enumerate(new_x):
        output[row_index, :] = interp1_inner(x, y, x_target)
    return output


def interp1_inner(x: np.ndarray, y: np.ndarray, new_x: float) -> np.ndarray:
    index = bisect_left_naive(x, new_x)
    print(f"Found {new_x} as position {index} in array")
    if index is None:
        out = np.ndarray((1, y.shape[1]))
        out[:] = np.nan
        return out
    x_interval = x[index + 1] - x[index]
    x_diff = new_x - x[index]
    ratio = x_diff / x_interval
    y_diff = y[index + 1, :] - y[index, :]
    return y[index, :] + ratio * y_diff


def bisect_left_naive(x_vector, x_target) -> int:
    def bisect_interval(x_vector_inner: np.ndarray, target: float):
        if len(x_vector_inner) == 2:
            if x_vector_inner[0] <= target < x_vector_inner[1]:
                return 0
            else:
                return None
        split_index = len(x_vector_inner) // 2
        split_value = x_vector_inner[split_index]
        if split_value > target:
            return bisect_interval(x_vector_inner[0 : split_index + 1], target)
        elif split_value == target:
            return split_index
        else:
            sub_index = bisect_interval(x_vector_inner[split_index:], target)
            return split_index + sub_index if sub_index is not None else None

    if len(x_vector) == 1:
        return 0
    if x_vector[-1] <= x_vector[0]:
        x_vector = -x_vector
        x_target = -x_target
    return bisect_interval(x_vector, x_target)


def to_fortran_string(number: float) -> str:
    return "{:.3e}".format(number).replace("e", "D")


def process_input(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        data_blocks = []
        flag = False
        block_index = 0
        for line in f.readlines():
            if int(line.lstrip()[0]) > 0:
                flag = True
                line_contents = [float(x) for x in line.split()]
                if len(data_blocks) <= block_index:
                    data_blocks.append(np.array(line_contents))
                else:
                    data_blocks[block_index] = np.vstack(
                        (data_blocks[block_index], line_contents)
                    )
            else:
                if flag:
                    block_index = block_index + 1
                    flag = False
    data_blocks[0] = data_blocks[0][-1:1:-1]
    return np.vstack(data_blocks)

def generate_arguments(lmbda: float, beta: float, eta: float, x0: float):
    return {"PAR": (("lambda", lmbda), ("beta", beta), ("epsil", eta)), "U": (("x", x0),)}, {}