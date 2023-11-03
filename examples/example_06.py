"""Example 06: Execution caching

Synopsis:
Caching taks execution depending on file existence.
"""
from pathlib import Path
import numpy as np
import ray

RESULT_DIR = Path("./results")
NUM_ITER = 5000000


# Result file name for input n
def output_fname(n):
    return f"case_{n}.npy"


@ray.remote
def foo(n):
    output_file = RESULT_DIR / output_fname(n)
    try:
        result = np.load(output_file)
        print(f"Results read from cache for file {output_file}")
        return result
    except OSError:
        print(f"File {output_file} not present, computing results.")
    rng = np.random.default_rng(n)
    mat = rng.normal(size=(30, 30))
    for _ in range(NUM_ITER):
        mat = (mat + mat) / 2
    np.save(output_file, mat)
    return mat


@ray.remote
def bar(mat):
    return mat.reshape(5, 6, 5, 6)


def main():
    ray.init()
    # Create directory for results if it does not exist.
    RESULT_DIR.mkdir(exist_ok=True)
    seeds = range(1, 9)
    refs = [bar.remote(foo.remote(i)) for i in seeds]
    result_parallel = np.sum(ray.get(refs), axis=0)
    print(f"Result shape: {result_parallel.shape}")


if __name__ == "__main__":
    main()
