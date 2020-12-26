import numpy as np
import tensorrt as trt
from aitrainer.common.trt_utils import load_engine, run_trt_engine


class SqueezeWave():
  def __init__(self):
    """SqueezeWave vocoder.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    self.engine = load_engine('models/trt/squeezewave.engine', TRT_LOGGER)
    self.context = self.engine.create_execution_context()

  def __call__(self, mel: np.ndarray) -> np.ndarray:
    """Convert mel features to an audio waveform using SqueezeWave.

    Args:
        mel (np.ndarray): The input mel features (e.g. from Tacotron2). Shape is [80, T_MEL].

    Returns:
        np.ndarray: The output audio waveform. Shape is [2 * T_MEL].
    """
    mel = mel[None, :].astype(np.float16)
    z = np.random.randn(1, 128, 2 * mel.shape[2]).astype(np.float16)
    audio = np.empty((1, 128 * 2 * mel.shape[2]), np.float16)

    tensors = {"inputs": {'mel': mel, 'z': z}, "outputs": {'audio': audio}}

    run_trt_engine(self.context, self.engine, tensors)

    return audio[0]
