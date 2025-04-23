import os
import torch
import re
import io
from collections import defaultdict

# === CONFIG ===
cfg_file = "crop_traj_test.txt"
ROOT = "D:\\Uni\\Masterarbeit"
LEADER = "Gello leader"
GRIPPER_FILE = "gripper_state.pt"

def parse_structure_file(file_path):
    tasks = defaultdict(lambda: defaultdict(list))
    current_task = None
    current_subtask = None
    current_episode = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Task line: [banana_from_sink_to_right_stove]
        task_match = re.match(r"\[([^\[\]]+)\]", line)
        if task_match:
            key = task_match.group(1)
            if "_" in key and "-" in key:  # Likely an episode
                current_episode = key
            elif current_episode:  # Inside images section
                if "[images]" in line:
                    ranges = re.findall(r"(\d+)\s*-\s*(\d+)", line)
                    for r in ranges:
                        start, end = map(int, r)
                        tasks[(current_task, current_subtask)][current_episode].append((start, end))
            else:
                # Could be a task or subtask
                if current_task is None:
                    current_task = key
                else:
                    current_subtask = key
            continue

        # Images line
        if "[images]" in line and current_episode:
            ranges = re.findall(r"(\d+)\s*-\s*(\d+)", line)
            for r in ranges:
                start, end = map(int, r)
                tasks[(current_task, current_subtask)][current_episode].append((start, end))

    return tasks

def print_gripper_states(tasks):
    for (task, subtask), episodes in tasks.items():
        print(f"\n--- Task: {task} | Subtask: {subtask} ---")
        for ep, ranges in episodes.items():
            print(f"\nEpisode: {ep}")
            ep_path = os.path.join(ROOT, task, subtask if subtask else '', ep, LEADER, GRIPPER_FILE)
            if not os.path.exists(ep_path):
                print(f"  [Gello leader] gripper_state.pt not found at: {ep_path}")
                continue

            data = torch.load(ep_path)

            for start, end in ranges:
                indices = list(range(start, end))  # last 5 before/at end
                print(f"  Range: {start}-{end} -> Indices used: {indices}")
                for i in indices:
                    if i < len(data):
                        print(f"    [{i}] {data[i]}")
                    else:
                        print(f"    [{i}] Index out of range!")

# === RUN ===
tasks = parse_structure_file(cfg_file)
print_gripper_states(tasks)
