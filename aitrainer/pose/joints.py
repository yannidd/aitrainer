import json

import numpy as np
from aitrainer.utils.math import mse


class JointAngleEstimator:
  def __init__(self, joints):
    self.joints = joints

    # Load human pose definition.
    with open('config/human_pose.json', 'r') as f:
      human_pose = json.load(f)

    self.keypoint_to_index = {kp: i for i, kp in enumerate(human_pose['keypoints'])}
    self.initial_angles = None

  @property
  def is_init(self):
    return (self.initial_angles is not None) and (not np.any(np.isnan(self.initial_angles)))

  def init(self, keypoints):
    initial_angles = self.estimate(keypoints)
    if self.initial_angles is None:
      self.initial_angles = initial_angles
    else:
      idxs = ~np.isnan(initial_angles)
      self.initial_angles[idxs] = initial_angles[idxs]

  def get_init_angles(self):
    return self.initial_angles if self.is_init else None

  def estimate(self, keypoints: dict) -> float:
    joints = self.joints
    keypoint_to_index = self.keypoint_to_index
    keypoint_coords = np.array(
        [[keypoints[keypoint] if keypoint in keypoints.keys() else [-1, -1] for keypoint in joint]
         for joint in joints])
    unavail_joints = np.any(keypoint_coords == -1, axis=(1, 2))
    # Recentre based on the vertex of the angle (e.g. move elbow to [0, 0]).
    rays1 = keypoint_coords[:, 0] - keypoint_coords[:, 1]
    rays2 = keypoint_coords[:, 2] - keypoint_coords[:, 1]
    # Turn into unit vectors.
    rays1 = rays1 / (np.linalg.norm(rays1, axis=1)[:, None] + 1e-9)
    rays2 = rays2 / (np.linalg.norm(rays2, axis=1)[:, None] + 1e-9)
    # Find the angles in the range [0, pi] as the arccos of the row-wise dot product.
    angles = np.arccos(np.einsum('ij,ij->i', rays1, rays2))
    angles[unavail_joints] = np.nan
    return angles

  def mse(self, keypoints):
    """The mean-squared error (MSE) between the initial and current joint angles.

    Args:
        keypoints (dict): A dictionnary with keypoints as keys and coordinates as values.

    Returns:
        [float]: The MSE.
    """
    if self.initial_angles is not None:
      joint_angles = self.estimate(keypoints)
      if not np.all(np.isnan(joint_angles)):
        return mse(joint_angles, self.initial_angles)
      else:
        return None
    else:
      raise Exception('Joint angles not initialised. Call JointAngleEstimator.init() first.')
