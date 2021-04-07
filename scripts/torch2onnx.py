# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
import argparse
import os
import sys

from aitrainer.tts.models.squeezewave.glow import SqueezeWaveForwardIsInfer, SqueezeWave


class SqueezeWaveOnnx(SqueezeWave):
  def infer_onnx(self, spect, z, sigma=1.0):
    spect_size = spect.size()
    l = spect.size(2) * (256 // self.n_audio_channel)

    # audio = torch.zeros(spect.size(0), self.n_remaining_channels, l).cuda().half()
    audio = z[:, :self.n_remaining_channels, :]

    for k in reversed(range(self.n_flows)):
      n_half = int(audio.size(1) // 2)
      audio_0 = audio[:, :n_half, :]
      audio_1 = audio[:, n_half:, :]

      output = self.WN[k]((audio_0, spect))
      s = output[:, n_half:, :]
      b = output[:, :n_half, :]
      audio_1 = (audio_1 - b) / torch.exp(s)
      audio = torch.cat([audio_0, audio_1], 1)

      audio = self.convinv[k](audio, reverse=True)

      if k % self.n_early_every == 0 and k > 0:
        # z = torch.zeros(spect.size(0), self.n_early_size, l).cuda().half()
        z_from = self.n_remaining_channels + (k // 2 - 1) * self.n_early_size
        z_to = z_from + self.n_early_size
        audio = torch.cat((sigma * z[:, z_from:z_to, :], audio), 1)

    audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
    return audio


def parse_args(parser):
  """
    Parse commandline arguments.
    """
  parser.add_argument('-o',
                      '--output',
                      type=str,
                      required=True,
                      help='Directory for the exported WaveGlow ONNX model')
  parser.add_argument('--fp16', action='store_true', help='inference with AMP')
  parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)

  return parser


def export_onnx(parser, args):

  ckpt = torch.load('models/squeezewave_l128_small.pth')
  squeezewave = SqueezeWaveOnnx(**ckpt['kwargs'])
  squeezewave.load_state_dict(ckpt['state_dict'])
  squeezewave = squeezewave.cuda()
  squeezewave = squeezewave.remove_weightnorm(squeezewave)
  squeezewave.forward = squeezewave.infer_onnx

  mel = torch.randn(1, 80, 20).cuda()
  z = torch.randn(1, 128, 2 * mel.size(2)).cuda()

  if args.fp16:
    mel = mel.half()
    z = z.half()
    squeezewave = squeezewave.half()

  a = squeezewave(mel, z)
  print(a.shape)

  torch.onnx.export(squeezewave, (mel, z),
                    'models/squeezewave.onnx',
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=["mel", "z"],
                    output_names=["audio"],
                    dynamic_axes={
                        "mel": {
                            0: "batch_size",
                            2: "mel_seq"
                        },
                        "z": {
                            0: "batch_size",
                            2: "z_seq"
                        },
                        "audio": {
                            0: "batch_size",
                            1: "audio_seq"
                        }
                    })

  # with torch.no_grad():
  #     # run inference to force calculation of inverses
  #     squeezewave.infer(mel, sigma=args.sigma_infer)

  #     # export to ONNX
  #     # if args.fp16:
  #     #     squeezewave = squeezewave.half()

  #     # squeezewave.forward = squeezewave.infer_onnx

  #     opset_version = 12

  #     output_path = "squeezewave.onnx"
  #     torch.onnx.export(squeezewave, (mel), output_path,
  #                       opset_version=opset_version,
  #                       do_constant_folding=True,
  #                       input_names=["mel"],
  #                       output_names=["audio"],
  #                       dynamic_axes={"mel":   {0: "batch_size", 2: "mel_seq"},
  #                                     # "z":     {0: "batch_size", 2: "z_seq"},
  #                                     "audio": {0: "batch_size", 1: "audio_seq"}})


def convert_resnet_pose_att():
  """Adapted from https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/trt_pose/utils/export_for_isaac.py
  """
  import argparse, json, os, re
  import torch
  import trt_pose.models
  from trt_pose.utils.export_for_isaac import InputReNormalization

  # Load model topology and define the model
  with open('config/human_pose.json', 'r') as f:
    topology = json.load(f)

  num_parts, num_links = len(topology['keypoints']), len(topology['skeleton'])
  model = trt_pose.models.MODELS['resnet18_baseline_att'](num_parts, num_links * 2).cuda().eval()

  # Load model weights
  model.load_state_dict(torch.load('models/torch/resnet18_pose_att.pth'))
  model = model.half()

  # Add InputReNormalization pre-processing and HeatmapMaxpoolAndPermute post-processing operations
  input_re_normalization = InputReNormalization().half()
  converted_model = torch.nn.Sequential(input_re_normalization, model).half()
  converted_model = model

  # Define input and output names for ONNX exported model.
  input_names = ["input"]
  output_names = ["heatmap", "part_affinity_fields"]

  # Export the model to ONNX.
  dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
  dummy_input = dummy_input.half()
  torch.onnx.export(converted_model,
                    dummy_input,
                    'models/onnx/resnet18_pose_att.onnx',
                    input_names=input_names,
                    output_names=output_names)
  print("Successfully completed convertion of %s to %s." %
        ('models/torch/resnet18_pose_att.pth', 'models/onnx/resnet18_pose_att.onnx'))


def main():

  # export_onnx(parser, args)
  convert_resnet_pose_att()


if __name__ == '__main__':
  main()
