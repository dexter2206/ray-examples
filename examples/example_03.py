"""Example 02: Running jobs in parallel, but with actual CPU load

The key observation to make here is that we are not pinning tasks
to CPU threads - it's the OS that does the scheduling.
"""
import numpy as np
import ray


NUM_ITER = 5000000


@ray.remote
def foo(n):
    rng = np.random.default_rng(n)
    mat = rng.normal(size=(30, 30))
    for _ in range(NUM_ITER):
        mat = (mat + mat) / 2
    return mat


def main():
    ray.init()

    seeds = range(1, 9)
    refs = [foo.remote(i) for i in seeds]
    result_parallel = np.sum(ray.get(refs), axis=0)
    print(f"Result shape: {result_parallel.shape}")


if __name__ == "__main__":
    main()
