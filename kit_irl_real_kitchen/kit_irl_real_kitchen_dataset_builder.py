import os
import cv2
from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class KitIrlRealKitchen(tfds.core.GeneratorBasedBuilder):
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
                        'image': tfds.features.Image(
                            shape=(250, 250, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(250, 250, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot joint state. Consists of [7x joint states]',
                        ),
                        'joint_state_velocity': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot joint velocities. Consists of [7x joint velocities]',
                        ),
                        'end_effector_pos': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='Current End Effector position in Cartesian space',
                        ),
                        'end_effector_ori': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='Current End Effector orientation in Cartesian space',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
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
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        data_path = "/home/marcelr/uha_test_policy/finetune_data/04_04_2024-15_53_21_0_17_79_banana_from_right_stove_to_sink/signal*.pickle"
        return {
            'train': self._generate_examples(path=data_path),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample, self._embed)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):
    data = {}
    path = os.path.join(episode_path, "*.pickle")
    for file in glob.glob(path):
        # Keys contained in .pickle:
        # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
        # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
        pickle_file_path = os.path.join(episode_path, file)
        data.update(np.load(pickle_file_path, allow_pickle=True))
    trajectory_length = data["traj_length"]
    cam1_path = os.path.join(episode_path, "cam_1")
    cam2_path = os.path.join(episode_path, "cam_2")
    cam1_image_vector = create_img_vector(cam1_path, trajectory_length)
    cam2_image_vector = create_img_vector(cam2_path, trajectory_length)
    data.update({'image': cam1_image_vector, 'wrist_image': cam2_image_vector})
    # print(data.keys())
    # print(data["image"][0].shape) # (250, 250, 3)
    # print(data["wrist_image"][0].shape) # (250, 250, 3)

    episode = []
    for i in range(trajectory_length):
        # compute Kona language embedding
        # language_embedding = embed([data['language_instruction']])[0].numpy() if embed is not None else np.zeros(512)
        language_embedding = embed(data['language_description']).numpy() if embed is not None else [np.zeros(512)]
        action = np.append(data['delta_end_effector_pos'][i], data['delta_end_effector_ori'][i], axis=0)
        action = np.append(action, data['des_gripper_width'][i])
        # action = data['delta_end_effector_pos'][i]
        # action = action.append(data['delta_end_effector_ori'][i])
        # action = action.append(data['des_gripper_width'][i])

        episode.append({
            'observation': {
                'image': data['image'][i],
                'wrist_image': data['wrist_image'][i],
                'joint_state': data['joint_state'][i],
                'joint_state_velocity': data['joint_state_velocity'][i],
                'end_effector_pos': data['end_effector_pos'][i],
                'end_effector_ori': data['end_effector_ori'][i],
            },
            'action': action,
            'action_joint_state': data['des_joint_state'][i],
            'action_joint_vel': data['des_joint_vel'][i],
            'action_gripper_width': data['des_gripper_width'][i],
            'delta_des_joint_state': data['delta_des_joint_state'][i],
            'discount': 1.0,
            'reward': float(i == (data['traj_length'] - 1)),
            'is_first': i == 0,
            'is_last': i == (data['traj_length'] - 1),
            'is_terminal': i == (data['traj_length'] - 1),
            'language_instruction': data['language_description'],
            'language_embedding': language_embedding,
        })

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path,
            'traj_length': data['traj_length'],
        }
    }

    # if you want to skip an example for whatever reason, simply return None
    return episode_path, sample

def create_img_vector(img_folder_path, trajectory_length):        
    cam_list = []
    cam_path_list = []
    for index in range(trajectory_length):
        frame_file_name = '{}.jpeg'.format(index)
        cam_path_list.append(frame_file_name)
        img_path = os.path.join(img_folder_path, frame_file_name)
        img_array = cv2.imread(img_path)
        cam_list.append(img_array)
    return cam_list

if __name__ == "__main__":
    data_path = "/home/marcelr/uha_test_policy/finetune_data/*"
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    # create list of all examples
    episode_paths = glob.glob(data_path)
    for episode in episode_paths:
        _, sample = _parse_example(episode, embed)
        print(sample)