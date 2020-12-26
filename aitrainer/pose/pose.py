import json
import logging
import time
from multiprocessing import Process, Queue, Value
from typing import Type

import numpy as np
import pycuda.driver as cuda
import torch
import trt_pose.coco
from aitrainer.camera.base import CameraBase
from aitrainer.pose.models.resnet import ResNet
from aitrainer.pose.utils import (KeypointDrawer, extract_keypoints, preprocess_image)
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
    cuda.init()
    device = cuda.Device(0)  # enter your Gpu id here
    ctx = device.make_context()

    # Load human pose definition and topology.
    with open('config/human_pose.json', 'r') as f:
      human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    keypoint_names = human_pose['keypoints']

    # Load trt pose model.
    logging.info('Loading the pose model...')
    resnet = ResNet()

    # Define mean and std for image preprocessing.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

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
      cmap, paf = resnet(data)
      cmap, paf = torch.Tensor(cmap[None, ...]), torch.Tensor(paf[None, ...])
      counts, objects, peaks = parse_objects(cmap, paf)
      keypoints = extract_keypoints(objects, peaks, keypoint_names)
      keypoint_queue.put(keypoints)

      if draw:
        keypoint_drawer.draw(image, objects, peaks)
  except KeyboardInterrupt:
    pass

  ctx.pop()
  camera.release()
