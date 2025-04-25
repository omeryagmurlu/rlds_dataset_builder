import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/mnikolaus/code/data/marc_rlds/kit_irl_real_kitchen_lang/1.0.0/kit_irl_real_kitchen_lang-train.tfrecord-00000-of-00001"
# file_path = "/home/mnikolaus/code/data/oemer_rlds/kit_irl_real_kitchen_lang-train.tfrecord-00000-of-00016"

# Load the TFRecord dataset
dataset = tf.data.TFRecordDataset(file_path)

amount_examples = 10

action_joint_states = []
joint_states = []

# Parse and extract joint states
for i, raw_record in enumerate(dataset.take(amount_examples)):  
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    for key, feature in example.features.feature.items():
        dtype = feature.WhichOneof('kind')
        value = np.array(getattr(feature, dtype).value)

        if key == "steps/action_joint_state":
            action_joint_states.append(value.reshape(-1, 7))  
        elif key == "steps/observation/joint_state":
            joint_states.append(value.reshape(-1, 7))  

# Convert lists to NumPy arrays
action_joint_states = np.vstack(action_joint_states)  # Flatten all episodes into one array
joint_states = np.vstack(joint_states)
print(action_joint_states.shape)
print(joint_states.shape)

# Plot Histogram for value distribution
plt.figure(figsize=(10, 5))
plt.hist(joint_states.flatten(), bins=100, alpha=0.7, color='blue', label="Joint State Values")
plt.xlabel("Joint Angle Value")
plt.ylabel("Frequency")
plt.title("Distribution of Joint States")
plt.legend()
plt.grid()
if "marc" in file_path:
    plt.savefig("out_dir/marc/joint_histogram.png")
elif "oemer" in file_path:
    plt.savefig("out_dir/oemer/joint_histogram.png")

# Plot Boxplot to see range and outliers
plt.figure(figsize=(10, 5))
plt.boxplot(joint_states, vert=True, patch_artist=True)
plt.xlabel("Joint Index")
plt.ylabel("Joint Value")
plt.title("Boxplot of Joint Values Across All Joints")
if "marc" in file_path:
    plt.savefig("out_dir/marc/joint_boxplot.png")
elif "oemer" in file_path:
    plt.savefig("out_dir/oemer/joint_boxplot.png")

# Plot a few joint states over time
plt.figure(figsize=(12, 6))
for i in range(min(7, joint_states.shape[1])):  # Plot first 7 joints
    plt.plot(joint_states[:, i], label=f"Joint {i}")
plt.xlabel("Timestep")
plt.ylabel("Joint Value")
plt.title("Joint State Progression Over Time")
plt.legend()
plt.grid()
if "marc" in file_path:
    plt.savefig("out_dir/marc/joint_progression.png")
elif "oemer" in file_path:
    plt.savefig("out_dir/oemer/joint_progression.png")
