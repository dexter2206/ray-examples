"""Example 04: setting CPU affinity.

Synopsis:
Running several tasks, in parallel on restricted number of cores, making shure that
each task runs on a single core.

Ray does not support setting CPU affinity, so we have to do this ourselves.
We do this by:

1. Restricting number of workers spawned by ray by setting num_cpu=N_CORES in ray.init
2. Mapping worker IDs to core numbers.
"""
import numpy as np
import psutil
import ray
from ray.util import state


# Number of cores to use
N_CORES = 3

# Assume we use cores 0, 1, 2
CORES = range(N_CORES)

# Number of iterations of each function (so that we get some real CPU load).
NUM_ITER = 5000000


# Get ID of the worker this process is executed in
def get_worker_ids():
    return [
        worker.worker_id
        for worker in state.list_workers()
        if worker.worker_type == "WORKER"
    ]


# Notice an additional argument affinity map, which is supposed to be
# a dictionary mapping worker id (str) to integer (core which should
# be used).
@ray.remote
def foo(n, affinity_map):
    # Get execution context, an object that provides meta information
    # about things like allocated resources, current node etc.
    # In particular it contains worker ID, which is what we're interested in.
    context = ray.get_runtime_context()
    # Map worker id from context to core
    core = affinity_map[context.get_worker_id()]
    # Set affinity
    psutil.Process().cpu_affinity([core])
    rng = np.random.default_rng(n)
    mat = rng.normal(size=(30, 30))

    for _ in range(NUM_ITER):
        mat = (mat + mat) / 2

    return mat


def main():
    # include_dashboard=True is crucial here, because otherwise ray does
    # not expose state API
    ray.init(num_cpus=N_CORES, include_dashboard=True)
    seeds = range(1, 9)

    # Get worker ids and map them to cores
    worker_ids = get_worker_ids()
    affinity_map = {wid: core for wid, core in zip(worker_ids, CORES)}

    # Remember to pass the affinity dictionary!
    refs = [foo.remote(i, affinity_map) for i in seeds]
    result_parallel = np.sum(ray.get(refs), axis=0)
    print(f"Result shape: {result_parallel.shape}")


if __name__ == "__main__":
    main()
