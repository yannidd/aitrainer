import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Compute the mean-squared error (MSE) between two arrays.

  If any values in b are np.nan, they are not considered in the calculation.

  Args:
      a (np.ndarray): The first array.
      b (np.ndarray): The second array.

  Returns:
      np.ndarray: The MSE - a scalar.
  """
  mask = np.bitwise_and(~np.isnan(a), ~np.isnan(b))
  normalizer = mask.sum() + 1e-9
  result = (np.square(a - b) * mask).mean(axis=None) / normalizer
  return result