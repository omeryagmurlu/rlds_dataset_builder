import os
import csv


def parse_dir(episode_path, lupus_csv_writer, lang_csv_writer):
    lupus_path = os.path.join(episode_path, "annotations", "lang_lupus.txt")
    lang_txt_path = os.path.join(episode_path, "lang.txt")
    if not os.path.isfile(lupus_path) or not os.path.isfile(lang_txt_path):
        return None
    
    with open(lupus_path, 'rb') as f:
        lupus_txt = f.read().decode("utf-8")

    with open(lang_txt_path, 'rb') as f:
        lang_txt = f.read().decode("utf-8")

    lupus_array = preprocess_string(lupus_txt)
    lang_array = preprocess_string(lang_txt)

    lupus_dict = {"file_name": episode_path}
    lang_dict = {"file_name": episode_path}
    lang_string = "language_instruction_"

    for i in range(len(lupus_array)):
        name = lang_string + str(i)
        lupus_dict[name] = lupus_array[i]

    for i in range(len(lang_dict)):
        name = lang_string + str(i)
        lang_dict[name] = lang_array[i]

    lupus_csv_writer.writerow(lupus_dict)
    lang_csv_writer.writerow(lang_dict)
        

def preprocess_string(unfiltered_str: str) -> list:
    lang_str = unfiltered_str[:unfiltered_str.find("\nconfidence:")]
    start = 0
    end = -1
    lang_array = []
    while True:
        end = lang_str[start:].find("\n")
        if end == -1:
            if len(lang_str[start:]) != 0:
                lang_array.append(lang_str[start:])
            break
        lang_array.append(lang_str[start:start + end])
        start += end + 1
      
    return lang_array

def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(full_path) if entry == "raw" else get_trajectorie_paths_recursive(full_path, sub_dir_list)
    # return subdirectories

if __name__ == "__main__":
    data_path = "/home/marcelr/BridgeData/raw"
    csv_path = "/home/marcelr/BridgeData"
    raw_dirs = []
    counter = 0
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    raw_dirs.reverse() # '/home/marcelr/BridgeData/raw/datacol1_toykitchen1/many_skills/09/2023-03-15_15-11-20/raw' '/home/marcelr/BridgeData/raw/datacol1_toykitchen1/many_skills/09/2023-03-15_15-11-20/raw'
    with open(os.path.join(csv_path, "lang_text.csv"), 'w', newline='') as lang_text_csv_file, open(os.path.join(csv_path, "lang_lupus.csv"), 'w', newline='') as lang_lupus_csv_file:
        fieldnames = ["file_name", "language_instruction_0", "language_instruction_1", "language_instruction_2", "language_instruction_3", "language_instruction_4", "language_instruction_5", "language_instruction_6",
                       "language_instruction_7", "language_instruction_8", "language_instruction_9", "language_instruction_10", "language_instruction_11", "language_instruction_12", "language_instruction_13", "language_instruction_14"]
        lupus_csv_writer = csv.DictWriter(lang_lupus_csv_file, delimiter=';', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        lang_csv_writer = csv.DictWriter(lang_text_csv_file, delimiter=';', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        lupus_csv_writer.writeheader()
        lang_csv_writer.writeheader()
        for raw_dir in raw_dirs:
            for traj_group in os.listdir(raw_dir):
                traj_group_full_path = os.path.join(raw_dir, traj_group)
                if os.path.isdir(traj_group_full_path):
                    for traj_dir in os.listdir(traj_group_full_path):
                        traj_dir_full_path = os.path.join(traj_group_full_path, traj_dir)
                        if os.path.isdir(traj_dir_full_path):
                            counter += 1
                            parse_dir(traj_dir_full_path, lupus_csv_writer, lang_csv_writer)
                            print(counter)
                        else:
                            print("non dir instead of traj found!")
                else:
                    print("non dir instead of traj_group found!")
