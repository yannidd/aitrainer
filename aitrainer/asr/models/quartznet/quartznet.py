import math
import warnings

import numpy as np
import tensorrt as trt
import torch
from aitrainer.asr.models.quartznet.values import FILTERBANKS, N_MEAN, N_STD
from aitrainer.common.trt_utils import load_engine, run_trt_engine
from scipy.special import softmax

warnings.filterwarnings('ignore', message='The function torch.rfft is deprecated and will be removed in a future PyTorch release. Use the new torch.fft module functions, instead, by importing torch.fft and calling torch.fft.fft or torch.fft.rfft.')  # yapf: disable


class QuartzNet:
  def __init__(self):
    """Converts a Mel features to token probabilities.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    self.engine = load_engine('models/trt/quartznet.engine', TRT_LOGGER)
    self.context = self.engine.create_execution_context()

  def __call__(self, mel: np.ndarray) -> np.ndarray:
    """Run QuartzNet.

    Converts a Mel features to token probabilities.

    Args:
        mel (np.ndarray): The Mel features of shape [64, N_MEL] and dtype np.float32.

    Returns:
        np.ndarray: Token probabilities of shape [⌈N_MEL / 2⌉, 29] and dtype np.float32.
    """
    mel = mel[None, :]
    logprobs = np.empty((1, int(math.ceil(mel.shape[2] / 2)), 29), np.float32)

    tensors = {"inputs": {'audio_signal': mel}, "outputs": {'logprobs': logprobs}}

    run_trt_engine(self.context, self.engine, tensors)

    return softmax(logprobs[0], 1)


class MelFeaturizer:
  def __init__(
      self,
      samplerate: int = 16000,
      n_fft: int = 512,
      n_mels: int = 64,
      hop_length: int = 160,
      win_length: int = 320,
      preemph: float = 0.97,
      n_mean: torch.Tensor = N_MEAN,
      n_std: torch.Tensor = N_STD,
  ):
    self.samplerate = samplerate
    self.n_fft = n_fft
    self.n_mels = n_mels
    self.hop_length = hop_length
    self.win_length = win_length
    self.preemph = preemph
    self.n_mean = n_mean
    self.n_std = n_std

    self.filterbanks = FILTERBANKS

    self.stft = lambda x: torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
        window=torch.hann_window(win_length, periodic=False),
        return_complex=False,
    )

  def __call__(self, x):
    x = torch.Tensor(x)
    x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
    # Disable autocast to get full range of stft values.
    with torch.cuda.amp.autocast(enabled=False):
      x = self.stft(x)
    x = torch.sqrt(x.pow(2).sum(-1))
    x = x.pow(2.0)
    x = torch.matmul(self.filterbanks.to(x.dtype), x)
    x = torch.log(x + 5.960464477539063e-08)

    if self.n_mean is not None and self.n_std is not None:
      x -= self.n_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)
      x /= self.n_std.view(x.shape[0], x.shape[1]).unsqueeze(2)
    return x[0].numpy()
