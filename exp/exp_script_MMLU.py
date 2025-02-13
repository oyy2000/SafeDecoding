import subprocess
from multiprocessing import Queue
import time
import threading
import json
import os


# File to store command execution logs
# Model, attacker, and defender options
MODEL_NAMES = ["mistral", "llama2"]  # "llama2"
ATTACKERS = ["MMLU", "HEx-PHI", "DeepInception"]  # "Just-Eval", "GCG", "AutoDAN", "PAIR",
# DEFENDERS = ["SafeDecoding", "Self-Exam", "no_defense"] # "LayerBugFixer" "Unlearning"
DEFENDERS = ["SafeDecoding", "PPL", "Paraphrase", "Retokenization", "Self-Reminder", "ICD", "Self-Exam", "no_defense"]  # "LayerBugFixer" "Unlearning"

MODEL_NAMES = ["mistral"]  # "llama2"
ATTACKERS = ["AdvBench", "HEx-PHI", "DeepInception", "Just-Eval"]  # "Just-Eval", "GCG", "AutoDAN", "PAIR",
# DEFENDERS = ["SafeDecoding", "Self-Exam", "no_defense"] # "LayerBugFixer" "Unlearning"
DEFENDERS = ["LayerBugFixer", "Unlearning"]  # "LayerBugFixer" "Unlearning"

MODEL_NAMES = ["llama2"]  # "llama2"
ATTACKERS = [ "MMLU"]  # "Just-Eval", "GCG", "AutoDAN", "PAIR",
DEFENDERS = ["SafeDecoding", "Paraphrase", "Self-Exam", "no_defense", "LayerBugFixer", "Unlearning"]  

MODEL_NAMES = ["mistral", "llama2"]  # "llama2"
ATTACKERS = [ "MMLU"]  # "Just-Eval", "GCG", "AutoDAN", "PAIR",
DEFENDERS = ["no_random_unlearning_1", "no_random_unlearning_2"]  

# DEFENDERS = ["LayerBugFixer", "Unlearning"]  # "LayerBugFixer" "Unlearning"


LOG_FILE = f"command_MODEL_NAMES_{MODEL_NAMES}_attackers_{ATTACKERS}_defenders_{DEFENDERS}.json"

def initialize_json_log(commands):
    """Initialize the JSON log file with all possible commands if not already initialized."""
    if not os.path.exists(LOG_FILE):
        logs = [{"command": cmd, "result": "pending"} for cmd in commands]
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)
        print(f"Initialized log file with {len(commands)} commands.")
    else:
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)

        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)
        print(f"Log file {LOG_FILE} already exists. Resuming with cleaned commands.")

def update_json_log(command, result):
    """Update the JSON log file with the command result."""
    lock = threading.Lock()
    with lock:
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)

        for log in logs:
            if log["command"] == command:
                log["result"] = result
                break

        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)

def load_pending_tasks():
    """Load tasks that are still pending or have failed from the log file."""
    with open(LOG_FILE, 'r') as f:
        logs = json.load(f)

    pending_tasks = [(i, log["command"]) for i, log in enumerate(logs)
                     if log["result"] == "pending" or log["result"].startswith("failed")]
    return pending_tasks

def get_gpu_count():
    """Returns the total number of GPUs available."""
    result = subprocess.run(['nvidia-smi', '--list-gpus'], stdout=subprocess.PIPE)
    gpu_count = len(result.stdout.decode('utf-8').strip().split('\n'))
    return gpu_count

def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    return [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

def get_gpu_total_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    return [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

def get_gpu_free_memory():
    usage = get_gpu_memory_usage()
    total = get_gpu_total_memory()
    free_memory = [t - u for t, u in zip(total, usage)]
    return free_memory

def run_command(gpu_id, command):
    process = subprocess.Popen(command, shell=True)
    return process

def monitor_gpu_usage(gpu_id, min_free_memory, timeout):
    start_time = time.time()
    while time.time() - start_time < timeout:
        free_memory = get_gpu_free_memory()[gpu_id]
        if free_memory >= min_free_memory:
            return True
        time.sleep(5)
    return False

def build_command(model_name, attacker, defender, disable_gpt_judge, gpt_api_key):
    if defender == "no_defense":
        return (
            f'python MMLU.py '
            f'--model_name {model_name} '
            f'--attacker {attacker} '
            f'--defense_off '
            f'{"--disable_GPT_judge" if disable_gpt_judge else ""} '
            f'--GPT_API {gpt_api_key}'
        ).strip()
    return (
        f'python MMLU.py '
        f'--model_name {model_name} '
        f'--attacker {attacker} '
        f'--defender {defender} '
        f'{"--disable_GPT_judge" if disable_gpt_judge else ""} '
        f'--GPT_API {gpt_api_key}'
    ).strip()
    
def worker(task_queue, gpu_id):
    min_free_memory = 20000  # Minimum required free memory in MB
    timeout = 120  # Timeout in seconds

    while True:
        task = task_queue.get()
        if task is None:
            break

        index, command = task

        while True:
            if monitor_gpu_usage(gpu_id, min_free_memory, timeout):
                full_command = f'CUDA_VISIBLE_DEVICES={gpu_id} {command}'
                print(f"Running on GPU {gpu_id}: {full_command}")
                process = run_command(gpu_id, full_command)
                update_json_log(command, "pending")

                exit_code = process.wait()
                if exit_code == 0:
                    print(f"Task {index} completed successfully.")
                    update_json_log(command, "success")
                else:
                    print(f"Task {index} failed with exit code {exit_code}.")
                    update_json_log(command, f"failed (exit code {exit_code})")
                time.sleep(20)
                break
            else:
                print(f"GPU {gpu_id} does not have enough free memory for task {index}. Waiting...")
                time.sleep(20)
def main():
    gpt_api_key = "sk"
    disable_gpt_judge = False

    task_queue = Queue()
    commands = []
    for model_name in MODEL_NAMES:
        for attacker in ATTACKERS:
            for defender in DEFENDERS:
                command = build_command(model_name, attacker, defender, disable_gpt_judge, gpt_api_key)
                commands.append(command)

    initialize_json_log(commands)
    pending_tasks = load_pending_tasks()
    if not pending_tasks:
        print("No pending tasks found. All tasks are completed.")
        return

    for task in pending_tasks:
        task_queue.put(task)

    gpu_count = get_gpu_count()

    # Signal workers to stop when tasks are done
    for _ in range(gpu_count):
        task_queue.put(None)

    workers = []
    for gpu_id in range(gpu_count):
        t = threading.Thread(target=worker, args=(task_queue, gpu_id))
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

if __name__ == "__main__":
    main()