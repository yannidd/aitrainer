import json
import sys
import time
from collections import deque
from threading import Thread

import colorlog as logging
import numpy as np
import torch
from aitrainer.camera.cv2 import CameraCv2
from aitrainer.pose.demo_mp import PoseEstimator
from aitrainer.tts.demo_mp import TTS
from aitrainer.utils.console import setup_logging
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


def countdown(from_value: int, tts: TTS = None):
  say = tts.say if tts is not None else logging.info
  say('Starting in...', block=True)
  for count in reversed(range(1, from_value + 1)):
    say(f'{count}.', block=True)
    time.sleep(0.5)
  say('Go!', block=True)


def count_reps(pose_estimator: PoseEstimator, tts: TTS, exercises: dict, exercise: str,
               count_to: int):
  joints = exercises[exercise]['joints']
  mse_threshold = exercises[exercise]['mse_threshold']
  time_based = exercises[exercise]['time_based']
  joint_angle_estimator = JointAngleEstimator(joints)
  counter = 0

  countdown_thread = Thread(target=countdown, args=(3, tts))
  countdown_thread.start()

  while countdown_thread.is_alive():
    joint_angle_estimator.init(pose_estimator.get_keypoints())

  if not joint_angle_estimator.is_init:
    logging.warning('Couldn\'t capture initial joint angles. Cancelling exercise.')
    return
  else:
    logging.info(f'Initial joint angles captured as {joint_angle_estimator.get_init_angles()}.')

  if time_based:
    raise NotImplementedError
  else:
    prev_mse = curr_mse = 0
    thresholded_hist = deque([1] * 3, maxlen=3)
    thresholded_hist_lp = deque([1] * 2, maxlen=2)
    while counter < count_to:
      if pose_estimator.keypoints_available:
        curr_mse = joint_angle_estimator.mse(pose_estimator.get_keypoints())
        curr_mse = curr_mse if curr_mse is not None else prev_mse
        prev_mse = curr_mse
        thresholded = int(curr_mse < mse_threshold)
        thresholded_hist.append(thresholded)
        if thresholded == 0:  # Filterring step.
          thresholded = 0 if sum(thresholded_hist) == 0 else 1
        thresholded_hist_lp.append(thresholded)
        counter += thresholded_hist_lp[-2] == 0 and thresholded_hist_lp[-1] == 1
        # print(thresholded_hist, end='\r')
        if thresholded_hist_lp[-2] == 0 and thresholded_hist_lp[-1] == 1:
          tts.say(f'{counter}.')


def main():
  # Setup logging.
  setup_logging()

  # Load exercises.
  with open('config/exercises.json', 'r') as f:
    exercises = json.load(f)

  pose_estimator = PoseEstimator(CameraCv2)
  pose_estimator.start()
  tts = TTS()
  tts.start()
  tts.say(f'Starting workout.', block=True)
  count_reps(pose_estimator, tts, exercises, 'dummy', 5)
  pose_estimator.stop()
  tts.stop()


if __name__ == '__main__':
  main()
