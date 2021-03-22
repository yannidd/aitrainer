import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import argparse
import tensorrt as trt
import os

import sys

from aitrainer.common.trt_utils import build_engine


def convert_squeezewave():
  shapes = [{
      "name": "mel",
      "min": (1, 80, 32),
      "opt": (1, 80, 768),
      "max": (1, 80, 1664)
  }, {
      "name": "z",
      "min": (1, 128, 2 * 32),
      "opt": (1, 128, 2 * 768),
      "max": (1, 128, 2 * 1664)
  }]
  onnx_path = 'models/squeezewave.onnx'
  engine_path = 'models/squeezewave.engine'
  print("Building SqueezeWave ...")
  squeezewave_engine = build_engine(onnx_path, shapes=shapes, fp16=True)
  if squeezewave_engine is not None:
    with open(engine_path, 'wb') as f:
      f.write(squeezewave_engine.serialize())
  else:
    print("Failed to build engine from", onnx_path)
    sys.exit()


def convert_tacotron2():
  encoder_onnx_path = 'models/tacotron2_encoder.onnx'
  encoder_engine_path = 'models/tacotron2_encoder.engine'
  decoder_iter_onnx_path = 'models/tacotron2_decoder_iter.onnx'
  decoder_iter_engine_path = 'models/tacotron2_decoder_iter.engine'
  postnet_onnx_path = 'models/tacotron2_postnet.onnx'
  postnet_engine_path = 'models/tacotron2_postnet.engine'

  # Encoder
  shapes = [{
      "name": "sequences",
      "min": (1, 4),
      "opt": (1, 128),
      "max": (1, 256)
  }, {
      "name": "sequence_lengths",
      "min": (1, ),
      "opt": (1, ),
      "max": (1, )
  }]
  print("Building Encoder ...")
  encoder_engine = build_engine(encoder_onnx_path, shapes=shapes, fp16=True)
  if encoder_engine is not None:
    with open(encoder_engine_path, 'wb') as f:
      f.write(encoder_engine.serialize())
  else:
    print("Failed to build engine from", encoder_onnx_path)
    sys.exit()

  # DecoderIter
  shapes = [{
      "name": "decoder_input",
      "min": (1, 80),
      "opt": (1, 80),
      "max": (1, 80)
  }, {
      "name": "attention_hidden",
      "min": (1, 1024),
      "opt": (1, 1024),
      "max": (1, 1024)
  }, {
      "name": "attention_cell",
      "min": (1, 1024),
      "opt": (1, 1024),
      "max": (1, 1024)
  }, {
      "name": "decoder_hidden",
      "min": (1, 1024),
      "opt": (1, 1024),
      "max": (1, 1024)
  }, {
      "name": "decoder_cell",
      "min": (1, 1024),
      "opt": (1, 1024),
      "max": (1, 1024)
  }, {
      "name": "attention_weights",
      "min": (1, 4),
      "opt": (1, 128),
      "max": (1, 256)
  }, {
      "name": "attention_weights_cum",
      "min": (1, 4),
      "opt": (1, 128),
      "max": (1, 256)
  }, {
      "name": "attention_context",
      "min": (1, 512),
      "opt": (1, 512),
      "max": (1, 512)
  }, {
      "name": "memory",
      "min": (1, 4, 512),
      "opt": (1, 128, 512),
      "max": (1, 256, 512)
  }, {
      "name": "processed_memory",
      "min": (1, 4, 128),
      "opt": (1, 128, 128),
      "max": (1, 256, 128)
  }, {
      "name": "mask",
      "min": (1, 4),
      "opt": (1, 128),
      "max": (1, 256)
  }]
  print("Building Decoder ...")
  decoder_iter_engine = build_engine(decoder_iter_onnx_path, shapes=shapes, fp16=True)
  if decoder_iter_engine is not None:
    with open(decoder_iter_engine_path, 'wb') as f:
      f.write(decoder_iter_engine.serialize())
  else:
    print("Failed to build engine from", decoder_iter_onnx_path)
    sys.exit()

  # Postnet
  shapes = [{"name": "mel_outputs", "min": (1, 80, 32), "opt": (1, 80, 768), "max": (1, 80, 1664)}]
  print("Building Postnet ...")
  postnet_engine = build_engine(postnet_onnx_path, shapes=shapes, fp16=True)
  if postnet_engine is not None:
    with open(postnet_engine_path, 'wb') as f:
      f.write(postnet_engine.serialize())
  else:
    print("Failed to build engine from", postnet_onnx_path)
    sys.exit()


def convert_pose():
  shapes = [{
      "name": "input",
      "min": (1, 3, 224, 224),
      "opt": (1, 3, 224, 224),
      "max": (1, 3, 224, 224)
  }]
  onnx_path = 'models/onnx/resnet18_pose_att.onnx'
  engine_path = 'models/trt/resnet18_pose_att.engine'
  print("Building ResNet18 ...")
  resnet_engine = build_engine(onnx_path, shapes=shapes, fp16=True)
  if resnet_engine is not None:
    with open(engine_path, 'wb') as f:
      f.write(resnet_engine.serialize())
  else:
    print("Failed to build engine from", onnx_path)
    sys.exit()


def convert_quartznet():
  shapes = [{
      "name": "audio_signal",
      "min": (1, 64, 247),  # 0.5 s
      "opt": (1, 64, 497),  # 1.0 s
      "max": (1, 64, 3997)  # 8.0 s
  }]
  onnx_path = 'models/quartznet.onnx'
  engine_path = 'models/quartznet.engine'
  print("Building QuartzNet ...")
  quartznet_engine = build_engine(onnx_path, shapes=shapes, fp16=False)
  if quartznet_engine is not None:
    with open(engine_path, 'wb') as f:
      f.write(quartznet_engine.serialize())
  else:
    print("Failed to build engine from", onnx_path)
    sys.exit()


convert_squeezewave()
convert_tacotron2()
convert_quartznet()