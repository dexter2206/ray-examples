"""Example 01: Basic ray execution

Synopsis:
We run (artificially long) jobs both serially and in parallel with ray.

Key takeways:
- You need to run ray.init() before submitting any tasks
- You decorate tasks with @ray.remote
- Tasks are scheduled by running f.remote
- f.remote does NOT return object but (possibly) schedules task for execution
- To obtain actual results (resolve task) run ray.get
"""
import ray
from time import sleep, time


# Usual Python function
def foo(n):
    sleep(n)
    return n**2


# Ray task. Does the same thing as the function above.
@ray.remote
def foo_ray(n):
    sleep(n)
    return n**2


def main():
    # Initialize ray
    ray.init()

    n_max = 10

    # Serial execution
    start = time()
    result_serial = sum(foo(i) for i in range(n_max))
    end = time()

    print(f"Serial result: {result_serial}, computed in {end-start}s")

    # Ray parallel execution
    start = time()
    # Observe, we don't use foo_ray(i) but foo_ray.remote(i)
    # Refs is a list of references to tasks results
    refs = [foo_ray.remote(i) for i in range(n_max)]
    # ray.get resolves given references. It may also be used on single reference
    # i.e. result = ray.get(foo_ray.remote(2))
    result_parallel = sum(ray.get(refs))
    end = time()
    # Since we are running in parallel, we expect execution time close to
    # the execution time of the longest job
    print(f"Parallel result: {result_parallel}, computed in {end-start}s")


if __name__ == "__main__":
    main()
