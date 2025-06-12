import GPUtil
from concurrent.futures import ThreadPoolExecutor
import itertools
import argparse
import time
import os


def refine(exp_name:str, view_dir: str, mask_type: str, joint: str, loss: str, steps: int, lr: float, device: str, seed: int, vis: bool):
    if vis:
        os.system(f"python joint_refinement.py --exp_name {exp_name} --view_dir {view_dir} --mask_type {mask_type} --joint {joint} --loss {loss} \
              --steps {steps} --lr {lr} --device {device} --seed {seed} --vis")
    else:
        os.system(f"python joint_refinement.py --exp_name {exp_name} --view_dir {view_dir} --mask_type {mask_type} --joint {joint} --loss {loss} \
                --steps {steps} --lr {lr} --device {device} --seed {seed}")
    
    
def worker(device_id: str, exp_name:str, view_dir: str, mask_type: str, joint: str, loss: str, steps: int, lr: float, seed: int, vis: bool):
    device = f"cuda:{device_id}"
    print(f"Starting job on {device} with {view_dir} on {joint} joint\n")
    refine(exp_name, view_dir, mask_type, joint, loss, steps, lr, device, seed, vis)
    print(f"Finished job on {device} with {view_dir} on {joint} joint\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--mask_type", type=str, choices=["gt", "monst3r"], required=True)
    parser.add_argument("--loss", type=str, choices=["chamfer", "hausdorff"], required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()

    exp_name_list = [args.exp_name]
    view_dir_list = [args.view_dir]
    mask_type_list = [args.mask_type]
    joint_type_list = ["revolute", "prismatic"]
    loss_list = [args.loss]
    steps_list = [args.steps]
    lr_list = [args.lr]
    seed_list = [args.seed]
    vis_list = [args.vis]
    jobs = list(itertools.product(exp_name_list, view_dir_list, mask_type_list, joint_type_list, loss_list, steps_list, lr_list, seed_list, vis_list))
    # Using ThreadPoolExecutor to manage the thread pool
    with ThreadPoolExecutor(max_workers=8) as executor:
        dispatch_jobs(jobs, executor)