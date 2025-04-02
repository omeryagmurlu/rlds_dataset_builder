import tensorflow as tf

file_path = "/home/mnikolaus/code/data/kit_irl_real_kitchen_lang-train.tfrecord-00000-of-00016"

out_dir = "/home/mnikolaus/code/data/pics_from_rlds"

# Load the TFRecord dataset
dataset = tf.data.TFRecordDataset(file_path)

# Parse and inspect the first few examples
for i, raw_record in enumerate(dataset.take(1)):  # Inspect first 3 examples
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    print(f"\nExample {i}:")
    for key, feature in example.features.feature.items():
        # Check the type of the feature
        dtype = feature.WhichOneof('kind')
        value = getattr(feature, dtype).value
        
        # Print key, dtype, and a sample of values
        # print(f"  - Key: {key}")
        if "image_top" in key:
            for idx, img in enumerate(value):
                with open(f"{out_dir}/image_top/img_{idx}.jpeg", "wb") as f:
                    f.write(img)
        if "image_side" in key:
            for idx, img in enumerate(value):
                with open(f"{out_dir}/image_side/img_{idx}.jpeg", "wb") as f:
                    f.write(img)
        # print(f"    Type: {dtype}")
        # print(f"    Values (first 5): {value[:5] if len(value) > 5 else value}")