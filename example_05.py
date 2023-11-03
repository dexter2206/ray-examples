"""Example 05: Chaining execution

Synopsis:
Running tasks that involve inputs from other tasks.
Importantly, we don't have to wait for the previous task to finish, and then
launch new tasks.

Key takeways:
You can pass object references as inputs to other ray functions and ray will
figure out they need to be actually resolved before being passed.
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


@ray.remote
def bar(mat):
    return mat.reshape(5, 6, 5, 6)


def main():
    ray.init()
    seeds = range(1, 9)
    # Observer that we don't need to run ray.get(foo.remote(i)) before
    # passing it to bar.remote.
    refs = [bar.remote(foo.remote(i)) for i in seeds]
    result_parallel = np.sum(ray.get(refs), axis=0)
    print(f"Result shape: {result_parallel.shape}")


if __name__ == "__main__":
    main()
