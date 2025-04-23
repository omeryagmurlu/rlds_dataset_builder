import os
import re

# Folders to skip
SKIP_FOLDERS = {"Gello leader", "Panda 102 follower"}

def get_image_range(folder):
    image_numbers = []
    for fname in os.listdir(folder):
        match = re.match(r"(\d+)\.png$", fname)
        if match:
            image_numbers.append(int(match.group(1)))
    if image_numbers:
        return min(image_numbers), max(image_numbers)
    return None, None

def print_structure(root, indent=0):
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or entry in SKIP_FOLDERS:
            continue

        if entry == "images":
            top_cam_path = os.path.join(path, "top_cam_orig")
            if os.path.isdir(top_cam_path):
                low, high = get_image_range(top_cam_path)
                if low is not None:
                    print(" " * indent + f"[images] {low} - {high}")
            continue

        print(" " * indent + f"[{entry}]")
        print_structure(path, indent + 2)


root_dir = "D:\\Uni\\Masterarbeit"
cfg_file = "crop_traj.txt"
print_structure(root_dir)
