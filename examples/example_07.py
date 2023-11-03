"""Example 06: Execution caching with decorator

Synopsis:
Caching taks execution depending on file existence.
This time we decouple caching logic from the actual computations.
"""
from pathlib import Path
import numpy as np
import ray

RESULT_DIR = Path("./results2")
NUM_ITER = 5000000


def foo_output_fname(n):
    return f"foo_case_{n}.npy"


# A decorator caching the function
# First level parameter: function which produces the file names.
def cache(output_fname_func):
    # Second function, this is actual decorator. func is the wrapped function.
    def _wrapper(func):
        # *args, **kwargs makes it possible to use the decorator with any function
        def _wrapped(*args, **kwargs):
            # Same logic as in previous example, but the computations are performed
            # by func.
            output_file = RESULT_DIR / output_fname_func(*args, **kwargs)
            try:
                result = np.load(output_file)
                print(
                    f"[{func.__name__}] Results read from cache for file {output_file}"
                )
                return result
            except OSError:
                print(f"File {output_file} not present, computing results.")
                result = func(*args, **kwargs)
                np.save(output_file, result)
                return result

        return _wrapped

    return _wrapper


# Decorator usage. Remember, it has to be innermost! (i.e. after ray.remote)
@ray.remote
@cache(foo_output_fname)
def foo(n):
    rng = np.random.default_rng(n)
    mat = rng.normal(size=(30, 30))
    for _ in range(NUM_ITER):
        mat = (mat + mat) / 2
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
