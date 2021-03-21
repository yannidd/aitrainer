import colorlog as logging
import numpy as np


def setup_logging(level: str = 'INFO'):
  """Setup logging format.


  Args:
      level (str, optional): Logging level. Defaults to 'INFO'.
  """
  np.set_printoptions(precision=2)
  logging.basicConfig(level=level,
                      format="%(log_color)s[%(levelname)-8s %(asctime)s] %(message)s",
                      datefmt='%H:%M:%S')
