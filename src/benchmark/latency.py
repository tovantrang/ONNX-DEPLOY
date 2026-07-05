import time

import numpy as np
from tqdm import tqdm


def measure_runner_latency(runner, dataloader, device, warmup=20, runs=100):
    timings = []
    iterator = iter(dataloader)
    for _ in range(warmup):
        x, _ = next(iterator)
        x = runner.prepare(x)
        runner.run(x)

    for _ in tqdm(range(runs)):
        x, _ = next(iterator)
        x = runner.prepare(x)
        start_time = time.perf_counter()
        runner.run(x)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

    timings = np.array(timings)
    return {
        "mean_latency_ms": timings.mean() * 1000,
        "p50_latency_ms": np.percentile(timings, 50) * 1000,
        "p95_latency_ms": np.percentile(timings, 95) * 1000,
    }
