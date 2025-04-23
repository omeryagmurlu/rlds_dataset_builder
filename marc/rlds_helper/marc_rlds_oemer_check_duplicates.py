import tensorflow as tf
file_path = "/home/mnikolaus/code/data/kit_irl_real_kitchen_lang-train.tfrecord-00000-of-00016"
out_dir = "/home/mnikolaus/code/data/pics_from_rlds"

# Load the TFRecord dataset
dataset = tf.data.TFRecordDataset(file_path)
amount_examples = 10
images = []

# Parse and inspect the first few examples
for i, raw_record in enumerate(dataset.take(amount_examples)):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    images_top = []
    images_side = []
    traj_length = None
    
    print(f"\nExample {i}:")
    for key, feature in example.features.feature.items():
        dtype = feature.WhichOneof('kind')
        if "image_top" in key:
            # For bytes_list, we need to access each image separately
            for img in getattr(feature, dtype).value:
                images_top.append(img)
        if "image_side" in key:
            for img in getattr(feature, dtype).value:
                images_side.append(img)
        if "traj_length" in key:
            traj_length = getattr(feature, dtype).value[0]  # Get first (and only) value
            print("length", traj_length)
    
    images.append({
        "images_top": images_top, 
        "images_side": images_side, 
        "length": traj_length
    })

# Check for duplicate images
for entry_idx, entry in enumerate(images):
    print(f"Checking example {entry_idx} with {len(entry['images_top'])} top images and {len(entry['images_side'])} side images")
    
    # Assuming your intention is to check consecutive frames
    for idx in range(1, len(entry["images_top"])):
        if entry["images_top"][idx] == entry["images_top"][idx - 1]:
            print(f"Example {entry_idx}: Duplicate top image at positions {idx-1} and {idx}")
        
    for idx in range(1, len(entry["images_side"])):
        if entry["images_side"][idx] == entry["images_side"][idx - 1]:
            print(f"Example {entry_idx}: Duplicate side image at positions {idx-1} and {idx}")