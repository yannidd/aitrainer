import cv2
import PIL.Image
import numpy as np


def extract_keypoints(objects, normalized_peaks, keypoint_names):
  keypoints = {}
  height = width = 224

  obj = objects[0][0]
  C = obj.shape[0]
  for j in range(C):
    k = int(obj[j])
    if k >= 0:
      peak = normalized_peaks[0][j][k]
      x = round(float(peak[1]) * width)
      y = round(float(peak[0]) * height)
      keypoints[keypoint_names[j]] = np.array([x, y])

  return keypoints


def bgr8_to_jpeg(value, quality=75):
  return bytes(cv2.imencode('.jpg', value)[1])


def preprocess_image(image, mean, std):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
  image = np.einsum('ijk->kij', image).copy()
  image -= mean[:, None, None]
  image /= std[:, None, None]
  return image.astype(np.float16)


class KeypointDrawer:
  def __init__(self, topology):
    self.topology = topology
    cv2.namedWindow('Pose Estimator', cv2.WINDOW_NORMAL)

  def draw(self, image, objects, normalized_peaks):
    topology = self.topology
    height = width = 224

    K = topology.shape[0]
    color = (0, 255, 0)
    obj = objects[0][0]
    C = obj.shape[0]
    for j in range(C):
      k = int(obj[j])
      if k >= 0:
        peak = normalized_peaks[0][j][k]
        x = round(float(peak[1]) * width)
        y = round(float(peak[0]) * height)
        cv2.circle(image, (x, y), 3, color, 2)

    for k in range(K):
      c_a = topology[k][2]
      c_b = topology[k][3]
      if obj[c_a] >= 0 and obj[c_b] >= 0:
        peak0 = normalized_peaks[0][c_a][obj[c_a]]
        peak1 = normalized_peaks[0][c_b][obj[c_b]]
        x0 = round(float(peak0[1]) * width)
        y0 = round(float(peak0[0]) * height)
        x1 = round(float(peak1[1]) * width)
        y1 = round(float(peak1[0]) * height)
        cv2.line(image, (x0, y0), (x1, y1), color, 2)

    image = cv2.flip(image, 1)
    cv2.imshow('Pose Estimator', image)
    cv2.waitKey(1)
