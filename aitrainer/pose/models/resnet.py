from typing import Tuple

import numpy as np
import tensorrt as trt
from aitrainer.common.trt_utils import load_engine, run_trt_engine


class ResNet:
  def __init__(self):
    """A pose model which takes an RGB image as an input and outputs a heatmap and part affinity 
    fields.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    self.engine = load_engine('models/trt/resnet18_pose_att.engine', TRT_LOGGER)
    self.context = self.engine.create_execution_context()

  def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run the pose model.

    Args:
        image (np.ndarray): The input image of shape [3, 224, 224]

    Returns:
        Tuple[np.ndarray, np.ndarray]: A heatmap [18, 56, 56] and part affinity fields 
        [42, 56, 56].
    """
    image = image[None, :]

    cmap = np.empty((1, 18, 56, 56), dtype=np.float16)
    paf = np.empty((1, 42, 56, 56), dtype=np.float16)

    tensors = {
        "inputs": {
            'input': image
        },
        "outputs": {
            'heatmap': cmap,
            'part_affinity_fields': paf,
        }
    }

    run_trt_engine(self.context, self.engine, tensors)

    return cmap[0], paf[0]
