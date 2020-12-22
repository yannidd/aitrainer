import logging
import os
import signal
import time
from multiprocessing import Process, Queue, Value

import numpy as np
import pyaudio
import torch
import torch.nn as nn
from aitrainer.tts.models.squeezewave.denoiser import Denoiser
from aitrainer.tts.models.squeezewave.glow import SqueezeWave
from aitrainer.tts.models.tacotron2.tacotron2 import Tacotron2
from aitrainer.tts.models.tacotron2.text import text_to_sequence
from scipy.io.wavfile import write

torch.set_grad_enabled(False)

MAX_WAV_VALUE = 32768.0


class TTS:
  def __init__(self):
    self.text_queue = Queue(maxsize=3)
    self.run = Value('i', 1)
    self.done_loading = Value('i', 0)
    self._is_speaking = Value('i', 0)
    self.process = Process(target=tts_worker,
                           args=(self.text_queue, self.run, self.done_loading, self._is_speaking))

  @property
  def is_speaking(self):
    return self._is_speaking.value

  def say(self, text, block=False):
    with self._is_speaking.get_lock():
      self._is_speaking.value = 1
    self.text_queue.put(text)
    while block and self.is_speaking:
      time.sleep(0.1)

  def start(self):
    logging.info('Starting TTS process...')
    self.process.start()
    while not self.done_loading.value:
      time.sleep(0.1)
    logging.info('TTS process finished starting.')

  def stop(self):
    with self.run.get_lock():
      self.run.value = 0
    self.process.join()


def say(text, squeezewave, tacotron2, denoiser, sd):
  text = ' ' + text + ' '
  logging.info(f'TTS saying: "{text.strip()}"')

  # preprocessing
  sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
  sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
  input_length = torch.IntTensor([sequence.size(1)]).long().cuda()

  # run the models
  with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequence, input_length)
    audio = squeezewave.infer(mel).float()
    audio = denoiser(audio)
    audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze().cpu().numpy().astype('int16')

  rate = 22050
  sd.play(audio, rate)
  sd.wait()


def tts_worker(text_queue: Queue, run: Value, done_loading: Value, is_speaking: Value):
  try:
    import sounddevice as sd

    # Load Tacotron2
    logging.info('Loading Tacotron2 for TTS...')
    ckpt = torch.load('models/tacotron2_fp32.pth')
    tacotron2 = Tacotron2(**ckpt['config'])
    tacotron2.load_state_dict(ckpt['state_dict'])
    tacotron2 = tacotron2.cuda().eval()
    tacotron2.text_to_sequence = text_to_sequence

    # Load SqueezeWave
    logging.info('Loading SqueezeWave for TTS...')
    ckpt = torch.load('models/squeezewave_l128_small.pth')
    squeezewave = SqueezeWave(**ckpt['kwargs'])
    squeezewave.load_state_dict(ckpt['state_dict'])
    squeezewave = squeezewave.cuda().eval()
    squeezewave = squeezewave.remove_weightnorm(squeezewave)

    # Create a Denoiser.
    logging.info('Creating a Denoiser for TTS...')
    denoiser = Denoiser(squeezewave).cuda()

    with done_loading.get_lock():
      done_loading.value = 1

    while run.value:
      if not text_queue.empty():
        text = text_queue.get()
        say(text, squeezewave, tacotron2, denoiser, sd)
      else:
        with is_speaking.get_lock():
          is_speaking.value = 0
      time.sleep(0.1)
  except KeyboardInterrupt:
    pass

