import logging
from typing import Tuple

import numpy as np
import tensorrt as trt
from aitrainer.common.trt_utils import load_engine, run_trt_engine
from aitrainer.tts.models.tacotron2.text import text_to_sequence
from scipy.special import expit as sigmoid


class Tacotron2:
  def __init__(self):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    self.encoder = load_engine('models/trt/tacotron2_encoder.engine', TRT_LOGGER)
    self.decoder_iter = load_engine('models/trt/tacotron2_decoder_iter.engine', TRT_LOGGER)
    self.postnet = load_engine('models/trt/tacotron2_postnet.engine', TRT_LOGGER)
    self.encoder_context = self.encoder.create_execution_context()
    self.decoder_iter_context = self.decoder_iter.create_execution_context()
    self.postnet_context = self.postnet.create_execution_context()

  def __call__(self, text: str) -> np.ndarray:
    # Preprocess the text.
    sequences, sequence_lengths = self._preprocess(text)

    # Encoder -------------------------------------------------------------------------------------
    memory = np.zeros((len(sequence_lengths), sequence_lengths[0], 512), dtype=np.float16)
    processed_memory = np.zeros((len(sequence_lengths), sequence_lengths[0], 128),
                                dtype=np.float16)
    lens = np.zeros_like(sequence_lengths)

    encoder_tensors = {
        "inputs": {
            'sequences': sequences,
            'sequence_lengths': sequence_lengths
        },
        "outputs": {
            'memory': memory,
            'lens': lens,
            'processed_memory': processed_memory
        }
    }

    run_trt_engine(self.encoder_context, self.encoder, encoder_tensors)

    # Decoder -------------------------------------------------------------------------------------
    mel_lengths = np.zeros([memory.shape[0]], dtype=np.int32)
    not_finished = np.ones([memory.shape[0]], dtype=np.int32)
    mel_outputs = np.zeros(1)
    gate_outputs = np.zeros(1)
    alignments = np.zeros(1)

    gate_threshold = 0.5
    max_decoder_steps = 1664
    first_iter = True

    decoder_inputs = self._init_decoder_inputs(memory, processed_memory, sequence_lengths)
    decoder_outputs = self._init_decoder_outputs(memory, sequence_lengths)

    measurements_decoder = {}
    while True:
      decoder_tensors = self._init_decoder_tensors(decoder_inputs, decoder_outputs)
      run_trt_engine(self.decoder_iter_context, self.decoder_iter, decoder_tensors)

      if first_iter:
        mel_outputs = decoder_outputs[7][:, :, None]
        gate_outputs = decoder_outputs[8][:, :, None]
        alignments = decoder_outputs[4][:, :, None]
        first_iter = False
      else:
        mel_outputs = np.concatenate((mel_outputs, decoder_outputs[7][:, :, None]), axis=2)
        gate_outputs = np.concatenate((gate_outputs, decoder_outputs[8][:, :, None]), axis=2)
        alignments = np.concatenate((alignments, decoder_outputs[4][:, :, None]), axis=2)

      dec = (sigmoid(decoder_outputs[8]) < gate_threshold).astype(np.int32)
      not_finished = not_finished * dec
      mel_lengths += not_finished[0]

      if np.sum(not_finished) == 0:
        break
      if mel_outputs.shape[2] == max_decoder_steps:
        logging.warning("Tacotron2 reached max decoder steps")
        break

      decoder_inputs, decoder_outputs = self._swap_inputs_outputs(decoder_inputs, decoder_outputs)

    # Postnet -------------------------------------------------------------------------------------
    mel_outputs_postnet = np.zeros_like(mel_outputs, dtype=np.float16)

    postnet_tensors = {
        "inputs": {
            'mel_outputs': mel_outputs
        },
        "outputs": {
            'mel_outputs_postnet': mel_outputs_postnet
        }
    }
    run_trt_engine(self.postnet_context, self.postnet, postnet_tensors)

    return mel_outputs_postnet[0]

  def _preprocess(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Encode strings to integers and pad the encoded sequences.

    Args:
        text (str): The text to be passed to Tacotron2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The encoded sequences and their lengths.
    """
    sequences = np.array(text_to_sequence(text, ['english_cleaners'])[:])[None, ...]
    sequence_lengths = np.array([len(text)])
    return sequences.astype('int32'), sequence_lengths.astype('int32')

  def _init_decoder_inputs(self, memory, processed_memory, memory_lengths):
    bs = memory.shape[0]
    seq_len = memory.shape[1]
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = np.zeros((bs, attention_rnn_dim), dtype=np.float16)
    attention_cell = np.zeros((bs, attention_rnn_dim), dtype=np.float16)
    decoder_hidden = np.zeros((bs, decoder_rnn_dim), dtype=np.float16)
    decoder_cell = np.zeros((bs, decoder_rnn_dim), dtype=np.float16)
    attention_weights = np.zeros((bs, seq_len), dtype=np.float16)
    attention_weights_cum = np.zeros((bs, seq_len), dtype=np.float16)
    attention_context = np.zeros((bs, encoder_embedding_dim), dtype=np.float16)
    mask = np.zeros(memory_lengths[0], dtype=bool)[None, ...]
    decoder_input = np.zeros((bs, n_mel_channels), dtype=np.float16)

    return (decoder_input, attention_hidden, attention_cell, decoder_hidden, decoder_cell,
            attention_weights, attention_weights_cum, attention_context, memory, processed_memory,
            mask)

  def _init_decoder_outputs(self, memory, memory_lengths):
    bs = memory.shape[0]
    seq_len = memory.shape[1]
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = np.zeros((bs, attention_rnn_dim), dtype=np.float16)
    attention_cell = np.zeros((bs, attention_rnn_dim), dtype=np.float16)
    decoder_hidden = np.zeros((bs, decoder_rnn_dim), dtype=np.float16)
    decoder_cell = np.zeros((bs, decoder_rnn_dim), dtype=np.float16)
    attention_weights = np.zeros((bs, seq_len), dtype=np.float16)
    attention_weights_cum = np.zeros((bs, seq_len), dtype=np.float16)
    attention_context = np.zeros((bs, encoder_embedding_dim), dtype=np.float16)
    decoder_output = np.zeros((bs, n_mel_channels), dtype=np.float16)
    gate_prediction = np.zeros((bs, 1), dtype=np.float16)

    return (attention_hidden, attention_cell, decoder_hidden, decoder_cell, attention_weights,
            attention_weights_cum, attention_context, decoder_output, gate_prediction)

  def _init_decoder_tensors(self, decoder_inputs, decoder_outputs):
    decoder_tensors = {
        "inputs": {
            'decoder_input': decoder_inputs[0],
            'attention_hidden': decoder_inputs[1],
            'attention_cell': decoder_inputs[2],
            'decoder_hidden': decoder_inputs[3],
            'decoder_cell': decoder_inputs[4],
            'attention_weights': decoder_inputs[5],
            'attention_weights_cum': decoder_inputs[6],
            'attention_context': decoder_inputs[7],
            'memory': decoder_inputs[8],
            'processed_memory': decoder_inputs[9],
            'mask': decoder_inputs[10]
        },
        "outputs": {
            'out_attention_hidden': decoder_outputs[0],
            'out_attention_cell': decoder_outputs[1],
            'out_decoder_hidden': decoder_outputs[2],
            'out_decoder_cell': decoder_outputs[3],
            'out_attention_weights': decoder_outputs[4],
            'out_attention_weights_cum': decoder_outputs[5],
            'out_attention_context': decoder_outputs[6],
            'decoder_output': decoder_outputs[7],
            'gate_prediction': decoder_outputs[8]
        }
    }
    return decoder_tensors

  def _swap_inputs_outputs(self, decoder_inputs, decoder_outputs):
    new_decoder_inputs = (
        decoder_outputs[7],  # decoder_output
        decoder_outputs[0],  # attention_hidden
        decoder_outputs[1],  # attention_cell
        decoder_outputs[2],  # decoder_hidden
        decoder_outputs[3],  # decoder_cell
        decoder_outputs[4],  # attention_weights
        decoder_outputs[5],  # attention_weights_cum
        decoder_outputs[6],  # attention_context
        decoder_inputs[8],  # memory
        decoder_inputs[9],  # processed_memory
        decoder_inputs[10])  # mask

    new_decoder_outputs = (
        decoder_inputs[1],  # attention_hidden
        decoder_inputs[2],  # attention_cell
        decoder_inputs[3],  # decoder_hidden
        decoder_inputs[4],  # decoder_cell
        decoder_inputs[5],  # attention_weights
        decoder_inputs[6],  # attention_weights_cum
        decoder_inputs[7],  # attention_context
        decoder_inputs[0],  # decoder_input
        decoder_outputs[8])  # gate_output

    return new_decoder_inputs, new_decoder_outputs
