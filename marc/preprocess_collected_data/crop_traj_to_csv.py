import csv
import re


tasks = ["[banana_from_sink_to_right_stove]", 
         "[banana_from_tray_to_right_stove]",
         "[bottle_from_right_stove_to_sink]",
         "[pot_from_right_to_left_stove]",
         "[pot_from_sink_to_right_stove]"]

subtasks = ["[let_go_of_pot_from_stove]", 
            "[move_to_left_stove_from_stove]", 
            "[align_above_right_stove_from_sink]", 
            "[border_of_sink(4x_different_pos)]"]

def build_range_pattern(n):
    if n < 1:
        raise ValueError("Must have at least one range")

    # Base pattern for a single "a - b"
    range_pattern = r'\d{1,3}\s*-\s*\d{1,3}'

    # First range (no comma before it)
    full_pattern = f'^{range_pattern}'

    # Add n-1 more ranges, each prefixed with a comma
    for _ in range(n - 1):
        full_pattern += r'\s*,\s*' + range_pattern

    # End of string
    full_pattern += '$'
    return full_pattern

def build_range_pattern_find(n):
    if n < 1:
        raise ValueError("Must have at least one range")

    # Base pattern for a single "a - b"
    range_pattern = r'(\d{1,3})\s*-\s*(\d{1,3})'

    # First range (no comma before it)
    full_pattern = f'^{range_pattern}'

    # Add n-1 more ranges, each prefixed with a comma
    for _ in range(n - 1):
        full_pattern += r'\s*,\s*' + range_pattern

    # End of string
    full_pattern += '$'
    return full_pattern


def parse_txt_to_csv(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    current_task = None
    current_subtasks = []
    current_episode = None
    current_pattern = None
    trajs = []
    path = [None, None]
    output_rows = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        print("stripped", stripped)
        if stripped in tasks:
            current_task = stripped[1:-1]
            path[0] = current_task
            continue
        
        if stripped =="[default_task]":
            current_subtasks = ["default_task"]
            path[1] = current_subtasks
            # Create regex pattern for "a - b, c - d" format where a, b, c, d are numbers between 0-999
            current_pattern = build_range_pattern(2)
            continue
        
        if stripped.startswith('[others]'):
            current_subtasks = stripped.split("[others] ")[1].strip().split(',')
            current_subtasks = [s.strip() for s in current_subtasks]
            path[1] = current_subtasks
            # Create regex pattern for "a - b, c - d" format where a, b, c, d are numbers between 0-999
            current_pattern = build_range_pattern(len(current_subtasks))
            continue
        
        
        if stripped.startswith('[') and stripped.endswith(']'):
            content = stripped[1:-1]
            # Check if content is in the format of YYYY_MM_DD-HH_MM_SS
            if re.match(r'^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$', content):
                current_episode = content
                continue
        
        if stripped.startswith('[images]'):
            traj = stripped.split("[images] ")[1].strip()
            # Check if the trajectory matches the pattern
            if re.match(current_pattern, traj):
                # Split the trajectory into two parts
                print("inside finding trajs")
                print("traj", traj)
                print("current_subtasks", current_subtasks)
                find_regex = build_range_pattern_find(len(current_subtasks) + 1)
                print("find_regex", find_regex)
                pairs = re.findall(find_regex, traj)
                print("pairs", pairs)
                # [('0', '82', '0', '74')]
                trajs = []
                for pair in range(int(len(pairs) / 2)):
                    trajs.append((pairs[pair][0], pairs[pair][1]))
                # [('0', '82'), ('0', '74')]
                print("trajs", trajs)
            else:
                print(f"Invalid trajectory format at line {i+1}: {line.strip()}")
            
        for j in range(len(current_subtasks)):
            print("j", j)
            # Append the task, subtask, episode, and trajectory to the output rows
            print("Appending to output rows")
            print(current_task)
            print(current_subtasks[j])
            print(current_episode)
            print(trajs)
            print(trajs[0][0])
            print(trajs[0][1])
            print(trajs[j+1][0])
            print(trajs[j+1][1])
            print(path[0])
            print(path[1][j])
            print("/".join([path[0], path[1][j], current_episode]))
            print(current_task, current_subtasks[j], current_episode, trajs[0][0], trajs[0][1], trajs[j+1][0], trajs[j+1][1], "/".join([path[0], path[1][j], current_episode]))
            output_rows.append([current_task, current_subtasks[j], current_episode, trajs[0][0], trajs[0][1], trajs[j+1][0], trajs[j+1][1], "/".join([path[0], path[1][j], current_episode])])
            
            
    # Write output to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['task', 'subtask', 'episode', 'original_start', 'original_end', 'cropped_traj_start', 'cropped_traj_end', 'path'])
        writer.writerows(output_rows)

# Run the parser
parse_txt_to_csv('crop_traj_test.txt', 'output.csv')
