import cv2
from aitrainer.camera.base import CameraBase


class CameraCv2(CameraBase):
  def __init__(self):
    self.vc = cv2.VideoCapture(0)
    self.vc.set(3, 224)
    self.vc.set(4, 224)

  def get_frame(self):
    frame = self.vc.read()[1]
    return frame[:224, 30:30 + 224, :]

  def release(self):
    self.vc.release()