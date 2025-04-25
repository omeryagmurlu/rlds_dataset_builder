import tensorflow as tf
from scipy.spatial.transform import Rotation
import torch

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

        if "image" in key:
            continue

        if "steps/action" == key or "steps/action_abs" == key or "ori" in key:
            # Print key, dtype, and a sample of values
            print(f"  - Key: {key}")
            print(f"    Type: {dtype}")
            print(f"    Values (first 7): {value[:7] if len(value) > 5 else value}")
            print(f"    Values shape: {len(value)}")
            # print(f"    Values: {(value)}")


# end effector ori x 3 2.88555908203125, 0.25256529450416565, 2.1694095134735107
# end effector ori quat x 4 0.4454444348812103, 0.8774592280387878, 0.053622420877218246, 0.16961489617824554
# action x 7 0.10955216735601425, -0.060164134949445724, -0.1638278216123581
# action abs x 7 2.7720048427581787, 0.1946772038936615, 1.9836630821228027

# action_ori = torch.tensor([0.10955216735601425, -0.060164134949445724, -0.1638278216123581], torch.float64)

eeo = torch.tensor([2.88555908203125, 0.25256529450416565, 2.1694095134735107], dtype=torch.float64)
eeoq = torch.tensor([0.4454444348812103, 0.8774592280387878, 0.053622420877218246, 0.16961489617824554], dtype=torch.float64)
action_abs_ori = torch.tensor([2.7720048427581787, 0.1946772038936615, 1.9836630821228027], dtype=torch.float64)

print("eeo", eeo)
print("trans to eeo", Rotation.from_quat(eeoq).as_euler("xyz"))
print(20 * "#")
print("eeoq", eeoq)
print("trans to eeoq", Rotation.from_euler("xyz", eeo).as_quat())