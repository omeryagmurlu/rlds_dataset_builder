import os
import cv2
from typing import Iterator, Tuple, Any
from scipy.spatial.transform import Rotation
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Bridge(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'depth_0': tfds.features.Image(
                            shape=(480, 640, 1),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='image of depth camera or padding 1s, if has_depth_0 is false.',
                        ),
                        'image_0': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of main camera or padding 1s, if has_image_0 is false.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of second camera or padding 1s, if has_image_0 is false.',
                        ),
                        'image_2': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of third camera or padding 1s, if has_image_0 is false.',
                        ),
                        'image_3': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of forth camera or padding 1s, if has_image_0 is false.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot end-effector state. Consists of [3x pos, 3x orientation (euler: roll, pitch, yaw), 1x gripper width]',
                        ),
                        'full_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot end-effector state. Consists of [3x pos, 3x orientation (euler: roll, pitch, yaw), 1x gripper width]',
                        ),
                        'desired_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot end-effector state. Consists of [3x pos, 3x orientation (euler: roll, pitch, yaw), 1x gripper width]',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Delta robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'new_robot_transform': tfds.features.Tensor(
                        shape=(4, 4),
                        dtype=np.float64,
                        doc='Field new_robot_transform from bridge dataset, probably some form of quat (x,y,z,w) in second dim'
                            'no information was given, cant check further'
                    ),
                    'delta_robot_transform': tfds.features.Tensor(
                        shape=(4, 4),
                        dtype=np.float64,
                        doc='Field delta_robot_transform from bridge dataset, probably some form of quat (x,y,z,w) in second dim'
                            'no information was given, cant check further'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction. uint8 encoded data from files, might need to be filtered (containes newline \n)'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(1, 512),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    )
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                    'traj_length': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples in trajectorie'
                    ),
                    'has_depth_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had a depth img, false if none (padding 1s in depth_0)'
                    ),
                    'has_image_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_0, false if none (padding 1s in image_0)'
                    ),
                    'has_image_1': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_1, false if none (padding 1s in image_1)'
                    ),
                    'has_image_2': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_2, false if none (padding 1s in image_2)'
                    ),
                    'has_image_3': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_3, false if none (padding 1s in image_3)'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had language annotations, false if none (empty string in language_instruction as padding)'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        data_path = "/home/marcelr/BridgeData/raw"
        return {
            'train': self._generate_examples(path=data_path),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # create list of all examples
        raw_dirs = []
        get_trajectorie_paths_recursive(path, raw_dirs)

        # for smallish datasets, use single-thread parsing
        for raw_dir in raw_dirs:
            for traj_group in os.listdir(raw_dir):
                traj_group_full_path = os.path.join(raw_dir, traj_group)
                if os.path.isdir(traj_group_full_path):
                    for traj_dir in os.listdir(traj_group_full_path):
                        traj_dir_full_path = os.path.join(traj_group_full_path, traj_dir)
                        if os.path.isdir(traj_dir_full_path):
                            yield _parse_example(traj_dir_full_path, self._embed)
                        else:
                            print("non dir instead of traj found!")
                            yield traj_dir_full_path, {}
                else:
                    print("non dir instead of traj_group found!")
                    yield traj_group_full_path, {}

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):
    data = {}

    for data_field in os.listdir(episode_path):
        data_field_full_path = os.path.join(episode_path, data_field)
        if os.path.isdir(data_field_full_path):
            cam1_image_vector = create_img_vector(data_field_full_path)
            data.update({data_field: cam1_image_vector})
        elif data_field == "lang.txt":
            with open(data_field_full_path, 'rb') as f:
                lang_txt = {"lang": f.read()}
                data.update(lang_txt)
        else:
            data.update({data_field[:data_field.find(".")]: np.load(data_field_full_path, allow_pickle=True)})

    # agent_data : dict_keys(['traj_ok', 'camera_info', 'term_t', 'stats'])
    # policy_out : dict_keys(['actions', 'new_robot_transform', 'delta_robot_transform', 'policy_type'])
    # obs_dict   : dict_keys(['joint_effort', 'qpos', 'qvel', 'full_state', 'state', 'desired_state', 'time_stamp', 'eef_transform', 'high_bound', 'low_bound', 'env_done', 't_get_obs', 'task_stage'])
    # lang.txt   : b'take the silver pot and place it on the top left burner\nconfidence: 1\n'
    # for key, value in data.items():
    #     print(key)
    #     if isinstance(value, list):
    #         print(value[0].keys())
    #     elif isinstance(value, dict):
    #         print(value.keys())
    #     else:
    #         print(value)

    trajectory_length = data["agent_data"]["term_t"]
    has_depth_0 = "depth_images0" in data
    has_image_0 = "images0" in data
    has_image_1 = "images1" in data
    has_image_2 = "images2" in data
    has_image_3 = "images3" in data
    has_language = "lang" in data

    pad_img_tensor = tf.ones([480, 640, 3], dtype=data["images0"][0].dtype).numpy()
    pad_depth_tensor = tf.ones([480, 640, 1], dtype=data["images0"][0].dtype).numpy()

    episode = []
    for i in range(trajectory_length):
        # compute Kona language embedding
        if embed is None:
            language_embedding = [np.zeros(512)]
        elif has_language:
            lang_str = lang_txt["lang"].decode("utf-8")
            lang_str = [lang_str[:lang_str.find("\n")]]
            language_embedding = embed(lang_str).numpy()
        else:
            language_embedding = embed("").numpy()

        episode.append({
            'observation': {
                "depth_0": data['depth_images0'][i] if has_depth_0 else pad_depth_tensor,
                "image_0": data['images0'][i] if has_image_0 else pad_img_tensor,
                "image_1": data['images1'][i] if has_image_1 else pad_img_tensor,
                "image_2": data['images2'][i] if has_image_2 else pad_img_tensor,
                "image_3": data['images3'][i] if has_image_3 else pad_img_tensor,
                "state": data["obs_dict"]["state"][i],
                "full_state": data["obs_dict"]["full_state"][i],
                "desired_state": data["obs_dict"]["desired_state"][i],
            },
            'action': data["policy_out"][i]["actions"],
            'new_robot_transform': data["policy_out"][i]["new_robot_transform"], # prbl quat, x,y,z,w
            'delta_robot_transform': data["policy_out"][i]["delta_robot_transform"], # prbl quat, x,y,z,w
            'discount': 1.0,
            'reward': float(i == (trajectory_length - 1)),
            'is_first': i == 0,
            'is_last': i == (trajectory_length - 1),
            'is_terminal': i == (trajectory_length - 1),
            'language_instruction': data['lang'] if has_language else b'',
            'language_embedding': language_embedding,
        })

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path,
            'traj_length': trajectory_length,
            'has_depth_0': has_depth_0,
            'has_image_0': has_image_0,
            'has_image_1': has_image_1,
            'has_image_2': has_image_2,
            'has_image_3': has_image_3,
            'has_language': has_language,
        }
    }

    # if you want to skip an example for whatever reason, simply return None
    return episode_path, sample

def create_img_vector(img_folder_path):
    cam_list = []
    cam_path_list = []
    for img_name in os.listdir(img_folder_path):
        cam_path_list.append(img_name)
        img_path = os.path.join(img_folder_path, img_name)
        img_array = cv2.imread(img_path)
        cam_list.append(img_array)
    return cam_list

def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(full_path) if entry == "raw" else get_trajectorie_paths_recursive(full_path, sub_dir_list)
    # return subdirectories

if __name__ == "__main__":
    data_path = "/home/marcelr/BridgeData/raw"
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    raw_dirs = []
    counter = 0
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    for raw_dir in raw_dirs:
        for traj_group in os.listdir(raw_dir):
            traj_group_full_path = os.path.join(raw_dir, traj_group)
            if os.path.isdir(traj_group_full_path):
                for traj_dir in os.listdir(traj_group_full_path):
                    traj_dir_full_path = os.path.join(traj_group_full_path, traj_dir)
                    if os.path.isdir(traj_dir_full_path):
                        counter += 1
                        _parse_example(traj_dir_full_path, embed)
                        print(counter)
                    else:
                        print("non dir instead of traj found!")
            else:
                print("non dir instead of traj_group found!")
    # create list of all examples
    # episode_paths = glob.glob(data_path)
    # for episode in episode_paths:
    #     _, sample = _parse_example(episode, embed)