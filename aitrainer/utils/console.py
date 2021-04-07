import colorlog as logging
import numpy as np


def setup_logging():
  """Setup logging format."""
  mpl_logger = logging.getLogger('matplotlib')
  mpl_logger.setLevel('WARNING') 
  np.set_printoptions(precision=2)
  logging.basicConfig(level='DEBUG',
                      format="%(log_color)s[%(levelname)-8s %(asctime)s] %(message)s",
                      datefmt='%H:%M:%S')