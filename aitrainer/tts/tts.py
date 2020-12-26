import logging
import time
from multiprocessing import Process, Queue, Value

import pycuda.driver as cuda
from aitrainer.tts.models.squeezewave import SqueezeWave
from aitrainer.tts.models.tacotron2 import Tacotron2

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

  # Run the models.
  mel = tacotron2(text)
  audio = (squeezewave(mel) * MAX_WAV_VALUE).astype('int16')

  # Play sound.
  rate = 22050
  sd.play(audio, rate)
  sd.wait()


def tts_worker(text_queue: Queue, run: Value, done_loading: Value, is_speaking: Value):
  try:
    import sounddevice as sd

    # Initialise CUDA.
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    # Load Tacotron2.
    logging.info('Loading Tacotron2 for TTS...')
    tacotron2 = Tacotron2()

    # Load SqueezeWave.
    logging.info('Loading SqueezeWave for TTS...')
    squeezewave = SqueezeWave()

    # Create a Denoiser.
    # TODO: Create a proper denoiser, currently it's just an identity function.
    logging.info('Creating a Denoiser for TTS...')
    denoiser = lambda x: x
    # denoiser = Denoiser(squeezewave).cuda()

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

  ctx.pop()
