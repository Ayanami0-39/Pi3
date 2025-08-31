import multiprocessing
import time
import random
from concurrent.futures import ProcessPoolExecutor
from itertools import count
from pi3.utils.geometry import se3_inverse, homogenize_points, depth_edge
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
import os
import sys
import torch

# Konfiguration
NUM_GPUS = 3
PROCESSING_TIME_MIN_MS = 50
PROCESSING_TIME_MAX_MS = 200
CLIENT_REQUEST_INTERVAL_MS = 30
TOTAL_REQUESTS = 20000


device_list = ["cuda:0", "cuda:1", "cuda:2"]
# model_list = [
#     torch.compile(Pi3.from_pretrained("yyfz233/Pi3").to(device_list[i]).eval())
#     for i in range(NUM_GPUS)
# ]
# change the above code to parallel load parameters

import torch, torch.multiprocessing as mp, os

model_dict = {}

def worker(rank, data):
    torch.cuda.set_device(rank)
    if model_dict.get(rank) is None:
        sys.stdout.write(f"Loading model on gpu-{rank}\n")
        sys.stdout.flush()
        model = torch.compile(Pi3.from_pretrained("yyfz233/Pi3").to(device_list[rank]).eval())
        model_dict[rank] = model
    model = model_dict[rank]
    data = data.to(device_list[rank])
    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            # output = model(data[None])
            times = []
            num_run = 10
            for _ in range(num_run):
                torch.cuda.synchronize()
                t0 = time.time()
                predictions = model(data[None])  # Add batch dimension
                torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)
            # Remove top 5 time diffs (slowest runs)
            times_sorted = sorted(times)
            if len(times_sorted) > 5:
                times_filtered = times_sorted[:-5]
            else:
                times_filtered = times_sorted
            avg_time = sum(times_filtered) / len(times_filtered)
            sys.stdout.write(f"GPU-{rank} Average inference time over {len(times_filtered)} runs (top 5 removed): {avg_time:.4f} seconds\n")
            sys.stdout.flush()
        # Process output
        sys.stdout.write(f"Processed data on gpu-{rank}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision('high')
    world_size = torch.cuda.device_count()
    target_dir = "/root/repos/Pi3/input_images_20250814_090447_171967"
    data_list = []
    for i in range(NUM_GPUS):
        imgs = load_images_as_tensor(os.path.join(target_dir, "images"), interval=1, PIXEL_LIMIT=255000).to(device_list[i])
        data_list.append(imgs)

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, data_list[rank]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print("All finished.")

