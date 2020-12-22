import json
import logging
import time
from multiprocessing import Process, Queue, Value
from multiprocessing.connection import Connection
from typing import Type

import torch
import torch2trt
import trt_pose.coco
from aitrainer.camera.base import CameraBase
from aitrainer.camera.cv2 import CameraCv2
from aitrainer.pose.utils import (KeypointDrawer, bgr8_to_jpeg, extract_keypoints,
                                  preprocess_image)
from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects


class PoseEstimator:
  def __init__(self, camera_class, draw=True):
    self.camera_class = camera_class
    self.draw = draw
    self.keypoint_queue = Queue(maxsize=10)
    self.run = Value('i', 1)
    self.done_loading = Value('i', 0)
    self.process = Process(target=pose_estimation_worker,
                           args=(self.keypoint_queue, self.run, self.done_loading,
                                 self.camera_class, self.draw))

  @property
  def keypoints_available(self):
    return not self.keypoint_queue.empty()

  def get_keypoints(self):
    return self.keypoint_queue.get()

  def start(self):
    logging.info('Starting PoseEstimator process...')
    self.process.start()
    while not self.done_loading.value:
      time.sleep(0.1)
    logging.info('PoseEstimator process finished starting.')

  def stop(self):
    with self.run.get_lock():
      self.run.value = 0
    self.process.join()


def pose_estimation_worker(keypoint_queue: Queue, run: Value, done_loading: Value,
                           camera_class: Type[CameraBase], draw: bool):
  try:
    # Load human pose definition and topology.
    with open('config/human_pose.json', 'r') as f:
      human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    keypoint_names = human_pose['keypoints']

    # Load trt pose model.
    logging.info('Loading the pose model...')
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('models/resnet18_baseline_att_trt.pth'))

    # Define mean and std for image preprocessing.
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    # Create object parser.
    parse_objects = ParseObjects(topology)

    # Create a camera instance.
    camera = camera_class()

    # Create a keypoint drawer if required.
    if draw:
      keypoint_drawer = KeypointDrawer(topology)

    with done_loading.get_lock():
      done_loading.value = 1

    while run.value:
      image = camera.get_frame()
      data = preprocess_image(image, mean, std)
      cmap, paf = model_trt(data)
      cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
      counts, objects, peaks = parse_objects(cmap, paf)
      keypoints = extract_keypoints(objects, peaks, keypoint_names)
      keypoint_queue.put(keypoints)

      if draw:
        keypoint_drawer.draw(image, objects, peaks)
  except KeyboardInterrupt:
    pass

  camera.release()
