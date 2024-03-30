#!/usr/bin/env python3
import argparse
import subprocess
import re
import time

def list_jobs(scheduler):
    """List jobs from the job scheduler."""
    if scheduler == "pbs":
        cmd = ["qstat"]
        job_id_index = 0
    elif scheduler == "slurm":
        cmd = ["squeue", "-o", "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"]
        job_id_index = 0
    else:
        raise ValueError("Unsupported scheduler")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split("\n")

    jobs = []
    for line in lines[2:]:  # Skip headers
        parts = line.split()
        jobs.append(parts[job_id_index])
    return jobs

def cancel_job(scheduler, job_id, attempts=2, delay=1):
    """Attempt to cancel a job with retries."""
    if scheduler == "pbs":
        cmd_base = ["qdel"]
    elif scheduler == "slurm":
        cmd_base = ["scancel"]
    else:
        raise ValueError("Unsupported scheduler")

    for _ in range(attempts):
        subprocess.run(cmd_base + [job_id])
        time.sleep(delay)  # Wait for a bit before checking
        if job_id not in list_jobs(scheduler):
            print(f"Job {job_id} canceled successfully.")
            return True
    print(f"Failed to cancel job {job_id} after {attempts} attempts.")
    return False

def parse_selection(selection, total_jobs):
    """Parse user selection into a list of indices."""
    selected_indices = set()
    parts = re.split(",| ", selection)
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            selected_indices.update(range(start, min(end + 1, total_jobs)))
        else:
            selected_indices.add(int(part))
    return sorted(selected_indices)

def main():
    print("Greetings. Cancellor is here to help 🧐")
    parser = argparse.ArgumentParser(description="""Job Management Utility for PBS and SLURM schedulers.""")
    parser.add_argument('--all', action='store_true', help='Delete all current jobs (with confirmation).')
    parser.add_argument('--attempts', type=int, default=2, help='Number of deletion attempts.')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay (in seconds) before reattempting deletion.')
    parser.add_argument('--scheduler', choices=['pbs', 'slurm'], help='Manually specify the scheduler type.')
    parser.add_argument('--which', type=str, help='Specify job NUMBERS to delete with syntax (e.g., 0-10, 21, 24 26).')
    args = parser.parse_args()

    scheduler = args.scheduler
    if not scheduler:
        scheduler = "pbs" if subprocess.run(["which", "qstat"], stdout=subprocess.PIPE).returncode == 0 else "slurm"

    jobs = list_jobs(scheduler)
    if not jobs:
        print("No jobs found.")
        greetings()
        return

    if args.all:
        confirm = input("Are you sure you want to delete all jobs? [y/N] ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
        for job_id in jobs:
            cancel_job(scheduler, job_id, args.attempts, args.delay)
    elif args.which:
        selected_indices = parse_selection(args.which, len(jobs))
        selected_jobs = [jobs[i] for i in selected_indices]
        for job_id in selected_jobs:
            cancel_job(scheduler, job_id, args.attempts, args.delay)
    else:
        for idx, job in enumerate(jobs):
            print(f"{idx} - {job}")
        selection = input("Enter job number(s) to cancel (e.g., 1, 3-5, all): ")
        if selection.lower() == "all":
            confirm = input("Are you sure you want to delete all jobs? [y/N] ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return
            selected_jobs = jobs
        else:
            selected_indices = parse_selection(selection, len(jobs))
            selected_jobs = [jobs[i] for i in selected_indices]
        for job_id in selected_jobs:
            cancel_job(scheduler, job_id, args.attempts, args.delay)
        print("Selected jobs have been processed for cancellation.")
        greetings()

def greetings():
    print("Greetings and au revoir 🎩")

if __name__ == "__main__":
    main()
