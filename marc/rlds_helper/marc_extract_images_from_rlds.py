import tensorflow as tf
from pathlib import Path

base_dir = Path("/home/mnikolaus/code/data/oemer_rlds")
out_dir = base_dir / "pot_from_right_to_left_stove"
out_dir.mkdir(parents=True, exist_ok=True)

id = 0

# Loop through all 16 tfrecord files
for file_idx in range(16):
    file_path = base_dir / f"rlds_files/kit_irl_real_kitchen_lang-train.tfrecord-000{file_idx:02d}-of-00016"
    dataset = tf.data.TFRecordDataset(str(file_path))
    
    print(f"Scanning file: {file_path}")
    
    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        for key, feature in example.features.feature.items():
            if "language_instr" in key:
                dtype = feature.WhichOneof('kind')
                value = getattr(feature, dtype).value
                instruction = value[0].decode("utf-8")
                
                if "pot" in instruction and "left" in instruction and "right" in instruction and "stove" in instruction:
                    print(f"Matched: {instruction}")
                    
                    sample_out_dir = out_dir / f"entry_{id}"
                    (sample_out_dir / "image_top").mkdir(parents=True, exist_ok=True)
                    (sample_out_dir / "image_side").mkdir(parents=True, exist_ok=True)
                    
                    # Save image_top
                    value = getattr(example.features.feature["steps/observation/image_top"], 
                                    example.features.feature["steps/observation/image_top"].WhichOneof('kind')).value
                    for idx, img in enumerate(value):
                        with open(sample_out_dir / "image_top" / f"img_{idx}.jpeg", "wb") as f:
                            f.write(img)
                    
                    # Save image_side
                    value = getattr(example.features.feature["steps/observation/image_side"], 
                                    example.features.feature["steps/observation/image_side"].WhichOneof('kind')).value
                    for idx, img in enumerate(value):
                        with open(sample_out_dir / "image_side" / f"img_{idx}.jpeg", "wb") as f:
                            f.write(img)

                    id += 1

                    break  # Only process each record once
