"""Example 02: Limiting resources.

Synopsis:
We run the same tasks as with example_01, but this time
we artificially require that each task requires 8 cpu cores.
This limits the parallelism, which should result in the increased execution times.

Key takeways:
- You can specify resources (num_cpus, num_gpus) in ray.remote decorator
- Number of CPUs is logical, i.e. it is your responsibility to make sure
  that tasks actually don't utilize more.
"""
import ray
from time import sleep, time


def foo(n):
    sleep(n)
    return n**2


# We declare that our task requires 8 cpu cores
@ray.remote(num_cpus=8)
def foo_ray(n):
    sleep(n)
    return n**2


def main():
    ray.init()

    n_max = 10

    # Serial execution
    start = time()
    result_serial = sum(foo(i) for i in range(n_max))
    end = time()

    print(f"Serial result: {result_serial}, computed in {end-start}s")

    # Ray parallel execution
    start = time()
    refs = [foo_ray.remote(i) for i in range(n_max)]
    result_parallel = sum(ray.get(refs))
    end = time()
    print(f"Parallel result: {result_parallel}, computed in {end-start}s")


if __name__ == "__main__":
    main()
