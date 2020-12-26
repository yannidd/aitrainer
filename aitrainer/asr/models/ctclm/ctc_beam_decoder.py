import os

import torch
from aitrainer.asr.models.quartznet.values import LABELS
from ctcdecode import CTCBeamDecoder


class Decoder:
  def __init__(
      self,
      labels: list = LABELS,
      beam_width: int = 100,
      model_path: str = None,
      alpha: float = 0.0,
      beta: float = 0.0,
      cutoff_top_n: int = 40,
      cutoff_prob: float = 1.0,
      blank_id: int = LABELS.index('_'),
      log_probs_input: bool = False,
  ):
    self.labels = labels
    self.beam_width = beam_width
    self.model_path = model_path
    self.alpha = alpha
    self.beta = beta
    self.cutoff_top_n = cutoff_top_n
    self.cutoff_prob = cutoff_prob
    self.blank_id = blank_id
    self.log_probs_input = log_probs_input

    self.decoder = CTCBeamDecoder(labels=labels,
                                  beam_width=beam_width,
                                  model_path=model_path,
                                  alpha=alpha,
                                  beta=beta,
                                  cutoff_top_n=cutoff_top_n,
                                  cutoff_prob=cutoff_prob,
                                  num_processes=max(os.cpu_count(), 1),
                                  blank_id=blank_id,
                                  log_probs_input=log_probs_input)

  def __call__(self, token_probs: torch.Tensor) -> str:
    """Generate a decoded string from token probabilities.

    Args:
        probs (torch.Tensor): The output from an acoustic model.

    Returns:
        str: The output string.
    """
    token_probs = torch.Tensor(token_probs[None, ...])
    beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(token_probs)
    tokens = beam_results[0][0]
    seq_len = out_lens[0][0]
    return ''.join([LABELS[x] for x in tokens[0:seq_len]])
