import os
import time

import numpy as np
import pyaudio
import sounddevice as sd
import torch
import torch.nn as nn
from aitrainer.tts.models.squeezewave.denoiser import Denoiser
from aitrainer.tts.models.squeezewave.glow import SqueezeWave
from aitrainer.tts.models.tacotron2.tacotron2 import Tacotron2
from aitrainer.tts.models.tacotron2.text import text_to_sequence
from scipy.io.wavfile import write

torch.set_grad_enabled(False)

MAX_WAV_VALUE = 32768.0

# Load Tacotron2
ckpt = torch.load('models/tacotron2_fp32.pth')
tacotron2 = Tacotron2(**ckpt['config'])
tacotron2.load_state_dict(ckpt['state_dict'])
tacotron2 = tacotron2.cuda().eval()
tacotron2.text_to_sequence = text_to_sequence

# Load SqueezeWave
ckpt = torch.load('models/squeezewave_l128_small.pth')
squeezewave = SqueezeWave(**ckpt['kwargs'])
squeezewave.load_state_dict(ckpt['state_dict'])
squeezewave = squeezewave.cuda().eval()
squeezewave = squeezewave.remove_weightnorm(squeezewave)

# Create a Denoiser.
denoiser = Denoiser(squeezewave).cuda()


def say(text):
  # preprocessing
  sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
  sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
  input_length = torch.IntTensor([sequence.size(1)]).long().cuda()

  # run the models
  with torch.no_grad():
    # print('Running Tacotron...')
    mel, _, _ = tacotron2.infer(sequence, input_length)
    print(mel.shape)
    # print('Running SqueezeWave...')
    audio = squeezewave.infer(mel).float()
    audio = denoiser(audio)
    audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze().cpu().numpy().astype('int16')

  rate = 22050
  sd.play(audio, rate)
  sd.wait()


try:
  while True:
    print('Say: ', end='')
    text = input()
    say(text)
except KeyboardInterrupt:
  pass
