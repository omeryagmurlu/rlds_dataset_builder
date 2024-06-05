import os
import csv


if __name__ == "__main__":
    csv_path = "/home/marcelr/BridgeData"
    output_dict = {}
    with open(os.path.join(csv_path, "lang_lupus.csv"), newline='') as csv_lupus, open(os.path.join(csv_path, "lang_text.csv"), newline='') as csv_text:
        reader_lupus = csv.DictReader(csv_lupus, delimiter=';')
        reader_text = csv.DictReader(csv_text, delimiter=';')
        for row_lupus, row_text in zip(reader_lupus, reader_text):
            if row_lupus["file_name"] != row_text["file_name"]:
                print("Error, non matching names!")
            else:
                if not row_text["language_instruction_0"] in output_dict:
                    output_dict[row_text["language_instruction_0"]] = []
                for i in range(15):
                    if row_lupus["language_instruction_" + str(i)] != "" and not row_lupus["language_instruction_" + str(i)] in output_dict[row_text["language_instruction_0"]]:
                        output_dict[row_text["language_instruction_0"]].append(row_lupus["language_instruction_" + str(i)])
                    else:
                        break

    max_length = 0
    from_label = ""
    for bridge_label, lupus_list in output_dict.items():
        curr_length = len(lupus_list)
        if curr_length > max_length:
            max_length = curr_length
            from_label = bridge_label
        if curr_length >= 100:
            print(bridge_label, " has", curr_length, " lupus labels")
    print("max length: ", max_length, " in label: ", from_label)

    with open(os.path.join(csv_path, "mapping.csv"), 'w', newline='') as mapping_csv_file:
        fieldnames = ["bridge_label", "lupus_label_0", "lupus_label_1", "lupus_label_2", "lupus_label_3", "lupus_label_4", "lupus_label_5", "lupus_label_6",
                      "lupus_label_7", "lupus_label_8", "lupus_label_9", "lupus_label_10", "lupus_label_11", "lupus_label_12", "lupus_label_13", "lupus_label_14",
                      "lupus_label_15", "lupus_label_16", "lupus_label_17", "lupus_label_18", "lupus_label_19", "lupus_label_20", "lupus_label_21", "lupus_label_22",
                    ]
        mapping_csv_writer = csv.DictWriter(mapping_csv_file, delimiter=';', quoting=csv.QUOTE_MINIMAL, fieldnames=fieldnames)
        mapping_csv_writer.writeheader()
        for bridge_label, lupus_list in output_dict.items():
            row_dict = {
                "bridge_label": bridge_label
            }
            for i in range(len(lupus_list)):
                row_dict["lupus_label_" + str(i)] = lupus_list[i]
                if i >= 22:
                    break
            mapping_csv_writer.writerow(row_dict)


