from aitrainer.asr import ASR
from aitrainer.utils.console import setup_logging


def main():
  setup_logging()

  asr = ASR(['mike'])

  asr.start()

  try:
    while True:
      if asr.is_available:
        asr.get_text()
  except KeyboardInterrupt:
    pass

  asr.stop()


if __name__ == "__main__":
  main()
