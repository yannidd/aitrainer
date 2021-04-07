import json
from collections import deque
from threading import Thread

import colorlog as logging

from aitrainer.camera.cv2 import CameraCv2
from aitrainer.pose import JointAngleEstimator, PoseEstimator
from aitrainer.utils.console import setup_logging
from aitrainer.utils.countdown import countdown


def count_reps(pose_estimator: PoseEstimator, exercises: dict, exercise: str, count_to: int):
  joints = exercises[exercise]['joints']
  mse_threshold = exercises[exercise]['mse_threshold']
  time_based = exercises[exercise]['time_based']
  joint_angle_estimator = JointAngleEstimator(joints)
  counter = 0

  countdown_thread = Thread(target=countdown, args=([3]))
  countdown_thread.start()

  while countdown_thread.is_alive():
    joint_angle_estimator.init(pose_estimator.get_keypoints())

  if not joint_angle_estimator.is_init:
    logging.warning(
        'Couldn\'t capture initial joint angles. Make sure you fit in the camera and the light is good. Cancelling exercise.'
    )
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
          logging.info(f'{counter}.')


def main():
  # Setup logging.
  setup_logging()

  # Load exercises.
  with open('config/exercises.json', 'r') as f:
    exercises = json.load(f)

  pose_estimator = PoseEstimator(CameraCv2)
  pose_estimator.start()
  count_reps(pose_estimator, exercises, 'squats', 10)
  pose_estimator.stop()


if __name__ == '__main__':
  main()
