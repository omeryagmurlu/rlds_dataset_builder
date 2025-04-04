import os
import cv2
from typing import Iterator, Tuple, Any
from scipy.spatial.transform import Rotation

import glob
import numpy as np
import natsort
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm
import torch
from pathlib import Path

tf.config.set_visible_devices([], "GPU")
data_path = "/home/hk-project-sustainebot/ob0961/ws_data/hkfswork/ob0961-data/data/flower_datasets/marc_rlds_test/collected_data"

class KitIrlRealKitchenLang(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_top': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main Top camera RGB observation.',
                        ),
                        'image_side': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Side camera RGB observation.',
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
                            doc='Current End Effector orientation in Cartesian space as Euler (xyz)',
                        ),
                        'end_effector_ori_quat': tfds.features.Tensor(
                            shape=(4,),
                            dtype=np.float64,
                            doc='Current End Effector orientation in Cartesian space as Quaternion',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Delta robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'action_abs': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Absolute robot action, consists of [3x end_effector_pos, '
                            '3x end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'action_joint_state': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_joint_vel': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action in joint space, consists of [7x joint velocities]',
                    ),
                    'delta_des_joint_state': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Delta robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_gripper_width': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Desired gripper width, consists of [1x gripper width] in range [0, 1]',
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
                    # """ 'language_instruction': tfds.features.Text(
                    #     doc='Language Instruction.'
                    # ),
                    # 'language_instruction_2': tfds.features.Text(
                    #     doc='Language Instruction.'
                    # ),
                    # 'language_instruction_3': tfds.features.Text(
                    #     doc='Language Instruction.'
                    # ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(3, 512),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ), """
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                    'traj_length': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples in trajectorie'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=data_path),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # create list of all examples
        raw_dirs = []
        get_trajectorie_paths_recursive(data_path, raw_dirs)
        print("# of trajectories:", len(raw_dirs))

        # for smallish datasets, use single-thread parsing
        for sample in raw_dirs:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):

    data = {}
    leader_path = os.path.join(episode_path, 'Gello leader/*.pt')
    follower_path = os.path.join(episode_path, 'Panda 102 follower/*.pt')

    # 'gripper_state.pt' 'ee_vel.pt' 'joint_vel.pt' 'ee_pos.pt' 'joint_pos.pt' (franka panda robot)
    for file in glob.glob(follower_path):
        name = Path(file).stem
        data.update({name : torch.load(file)})
    # 'gripper_state.pt' 'joint_pos.pt' (gello)
    for file in glob.glob(leader_path):
        name = 'des_' + Path(file).stem
        data.update({name : torch.load(file)})
    # all data entry should have the same traj length (primary length of tensor)
    trajectory_length = data[list(data.keys())[0]].size()[0]


    top_cam_path = os.path.join(episode_path, 'images/top_cam_processed')
    side_cam_path = os.path.join(episode_path, 'images/side_cam_processed')
    top_cam_vector = create_img_vector(top_cam_path, trajectory_length)
    side_cam_vector = create_img_vector(side_cam_path, trajectory_length)

    data.update({
                'image_top': top_cam_vector, 
                'image_side': side_cam_vector
                })

    episode = []

    delta_ee_pos = np.zeros((data["ee_pos"].size()[0], 7))
    delta_des_joint_state = torch.zeros_like(data["des_joint_pos"])

    for i in range(trajectory_length):

        # compute deltas: delta_ee_pos (pos and ori) & delta_des_joint_state
        if i == 0:
            delta_ee_pos[i] = 0
            delta_des_joint_state[i] = 0
        else:
            delta_ee_pos[i][0:3] = data["ee_pos"][i][0:3] - data["ee_pos"][i-1][0:3]

            rot_quat = Rotation.from_quat(data["ee_pos"][i][3:])
            rot_quat_prev = Rotation.from_quat(data["ee_pos"][i-1][3:7])
            delta_rot = rot_quat * rot_quat_prev.inv()
            delta_ee_pos[i][3:6] = delta_rot.as_euler("xyz")

            delta_des_joint_state[i] = data["des_joint_pos"][i] - data["des_joint_pos"][i-1]


        # compute action 
        action_pos_ori = delta_ee_pos[i][0:6]
        action_gripper = data['des_gripper_state'][i]
        action = np.append(action_pos_ori, action_gripper)
        # compute action_abs
        action_abs_pos = data["ee_pos"][i][0:3]
        action_abs_ori = Rotation.from_quat(data["ee_pos"][i][3:7]).as_euler("xyz")
        action_abs_gripper = data['des_gripper_state'][i]
        action_abs = np.concatenate([action_abs_pos, action_abs_ori, [action_abs_gripper]])

        # compute eef orientation
        end_effector_ori = Rotation.from_quat(data["ee_pos"][i][3:7]).as_euler("xyz")

        episode.append({
            'observation': {
                'image_top': data['image_top'][i],
                'image_side': data['image_side'][i],
                'joint_state': data['joint_pos'][i],
                'joint_state_velocity': data['joint_vel'][i],
                'end_effector_pos': data['ee_pos'][i][0:3],
                'end_effector_ori_quat': data['ee_pos'][i][3:7], 
                'end_effector_ori': end_effector_ori,
            },
            'action': action,
            'action_abs': action_abs,
            'action_joint_state': data['des_joint_pos'][i],
            'action_joint_vel': data['joint_vel'][i], # 'action_joint_vel': data['des_joint_vel'][i],
            'action_gripper_width': data['des_gripper_state'][i],
            'delta_des_joint_state': delta_des_joint_state[i],
            'discount': 1.0,
            'reward': float(i == (trajectory_length - 1)),
            'is_first': i == 0,
            'is_last': i == (trajectory_length - 1),
            'is_terminal': i == (trajectory_length - 1),
            # 'language_instruction': data['language_description'][0],
            # 'language_instruction_2': data['language_description'][1],
            # 'language_instruction_3': data['language_description'][2],
            # 'language_embedding': language_embedding,
        })

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path,
            'traj_length': trajectory_length,
        }
    }

    # if you want to skip an example for whatever reason, simply return None
    return episode_path, sample

def create_img_vector(img_folder_path, trajectory_length):
    cam_list = []
    img_paths = glob.glob(os.path.join(img_folder_path, '*.jpeg'))
    img_paths = natsort.natsorted(img_paths)
    assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    for img_path in img_paths:
        img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        cam_list.append(img_array)
    return cam_list

def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(directory) if entry == "images" else get_trajectorie_paths_recursive(full_path, sub_dir_list)

if __name__ == "__main__":
    # create list of all examples
    raw_dirs = []
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    for trajectorie_path in tqdm(raw_dirs):
        _, sample = _parse_example(trajectorie_path)
        # print(f"sample: {sample}")