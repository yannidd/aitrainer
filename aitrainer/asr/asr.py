import logging
import os
import time
import warnings
from multiprocessing import Process, Queue, Value
from threading import Thread

import numpy as np
import pycuda.driver as cuda
import torch
from aitrainer.asr.models.ctclm import Decoder
from aitrainer.asr.models.quartznet import MelFeaturizer, QuartzNet
from aitrainer.utils.console import setup_logging
from scipy.io import wavfile

warnings.filterwarnings('ignore', message='The function torch.rfft is deprecated and will be removed in a future PyTorch release. Use the new torch.fft module functions, instead, by importing torch.fft and calling torch.fft.fft or torch.fft.rfft.')  # yapf: disable


class ASR:
  def __init__(self, activation_words):
    self.text_queue = Queue(maxsize=1)
    self.run = Value('i', 1)
    self.done_loading = Value('i', 0)
    self.process = Process(target=asr_worker, args=(self.text_queue, self.run, self.done_loading))

  @property
  def is_available(self):
    return not self.text_queue.empty()

  def get_text(self):
    return self.text_queue.get()

  def start(self):
    logging.info('Starting ASR process...')
    self.process.start()
    while not self.done_loading.value:
      time.sleep(0.1)
    logging.info('ASR process finished starting.')

  def stop(self):
    with self.run.get_lock():
      self.run.value = 0
    self.process.join()


def asr_worker(text_queue: Queue, run: Value, done_loading: Value):
  try:
    import sounddevice as sd
    import soundfile as sf

    # Initialise CUDA.
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    # Load the QuartzNet ASR model.
    logging.info('Loading QuartzNet model for ASR...')
    featurizer = MelFeaturizer()
    quartznet = QuartzNet()

    # Initialise the Decoder.
    logging.info('Loading CTC Beam Decoder...')
    decoder = Decoder(model_path='models/lm/3_gram_lm.trie', alpha=1, beta=0.5)

    with done_loading.get_lock():
      done_loading.value = 1

    chunk_size = 1 * 16000
    n_past_chunks = 5
    past_chunks_size = chunk_size * (n_past_chunks - 1)
    activation_words = ['jarvis', 'jervis']
    beep, _ = sf.read('assets/wav/beep.wav', dtype='float32')
    peeb = np.ascontiguousarray(np.flip(beep))
    activation_waveform = np.zeros((n_past_chunks * chunk_size, 1), dtype=np.float32)
    in_stream = sd.InputStream(samplerate=16000, channels=1)
    out_stream = sd.OutputStream(samplerate=44100, channels=2)
    in_stream.start()
    out_stream.start()

    while run.value:
      # Read waveform from the microphone and store in the rolling buffer.
      data, overflowed = in_stream.read(chunk_size)
      if overflowed:
        logging.warning('ASR process is skipping microphone frames!')
      activation_waveform = np.roll(activation_waveform, -chunk_size)
      activation_waveform[past_chunks_size:, 0] = data[:, 0]
      # Run ASR.
      token_probs = quartznet(featurizer(activation_waveform.T))
      decoded = decoder(token_probs)
      # If the keyword was said...
      if any([word in decoded for word in activation_words]):
        logging.info('ASR triggered!')
        # Play a beep sound.
        out_stream.write(beep)
        # Read waveform from the microphone.
        _data = in_stream.read(5 * 16000)[0]
        # Play a peeb sound.
        out_stream.write(peeb)
        # Run ASR.
        token_probs = quartznet(featurizer(_data.T))
        decoded = decoder(token_probs)
        # Add the recognised text to the text queue and reset the activation waveform buffer.
        logging.info(f'ASR recognised: "{decoded}".')
        text_queue.put(decoded)
        activation_waveform *= 0
  except KeyboardInterrupt:
    pass

  ctx.pop()
  in_stream.stop()
  out_stream.stop()
