import os
import torch
import re
import io
from collections import defaultdict

# === CONFIG ===
ROOT = "/home/hk-project-sustainebot/ob0961/ws_data/hkfswork/ob0961-data/data/marc_collected_data/marc_datasets"
cfg_file = "/home/hk-project-sustainebot/ob0961/repos/rlds_dataset_builder/marc/preprocess_collected_data/crop_traj_test.txt"
LEADER = "Gello leader"
GRIPPER_FILE = "gripper_state.pt"


def parse_structure_file(file_path):
    tasks = defaultdict(lambda: defaultdict(list))
    current_task = None
    current_subtasks = []  # support multiple
    current_episode = None
    multi_subtask_mode = False

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Task or metadata line: [banana_from_sink_to_right_stove]
        task_match = re.match(r"\[([^\[\]]+)\]", line)
        if task_match:
            key = task_match.group(1)
            if "_" in key and "-" in key:  # Likely an episode
                current_episode = key
            elif current_episode:  # Likely entering [images] block
                continue
            else:
                if current_task is None:
                    current_task = key
                elif key == "others":
                    # Multi-subtask mode
                    multi_subtask_mode = True
                    continue  # will parse subtasks next
                else:
                    current_subtasks = [key]
                    multi_subtask_mode = False
            continue

        # Line with subtasks after [others]
        if multi_subtask_mode and ',' in line:
            current_subtasks = [s.strip() for s in line.split(',')]
            continue

        # Images line
        if "[images]" in line and current_episode:
            ranges = re.findall(r"(\d+)\s*-\s*(\d+)", line)

            # Skip the first range in multi-subtask mode
            if multi_subtask_mode:
                ranges = ranges[1:]

            for idx, r in enumerate(ranges):
                start, end = map(int, r)
                subtask = current_subtasks[idx] if idx < len(current_subtasks) else f"subtask_{idx}"
                tasks[(current_task, subtask)][current_episode].append((start, end))

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
