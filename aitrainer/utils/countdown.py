import time

import colorlog as logging


def countdown(from_value: int):
  logging.info('Starting in...')
  for count in reversed(range(1, from_value + 1)):
    logging.info(f'{count}.')
    time.sleep(1.0)
  logging.info('Go!')
