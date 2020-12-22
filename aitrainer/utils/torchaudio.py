"""
Import torchaudio, but suppress "sox" backend deprecation warning.
"""
import warnings
warnings.filterwarnings('ignore', message='"sox" backend is being deprecated. The default backend will be changed to "sox_io" backend in 0.8.0 and "sox" backend will be removed in 0.9.0. Please migrate to "sox_io" backend. Please refer to https://github.com/pytorch/audio/issues/903 for the detail.')  # yapf: disable
import torchaudio