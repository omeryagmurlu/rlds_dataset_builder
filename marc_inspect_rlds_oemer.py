import tensorflow as tf

file_path = "/home/hk-project-sustainebot/ob0961/ws_data/hkfswork/ob0961-data/data/flower_datasets/marc_rlds_test/kit_irl_real_kitchen_lang/1.0.0kit_irl_real_kitchen_lang-train.tfrecord-00000-of-00016"

# action_joint_state 0:7
# joint_state 0:7

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
        print(f"  - Key: {key}")
        print(f"    Type: {dtype}")
        # print(f"    Values (first 5): {value[:5] if len(value) > 5 else value}")