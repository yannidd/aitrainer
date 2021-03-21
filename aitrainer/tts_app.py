from aitrainer.tts import TTS
from aitrainer.utils.console import setup_logging


def main():
  setup_logging()

  tts = TTS()  # Text-to-speech engine.
  tts.start()  # Start the TTS process.

  # Get input from the terminal and generate speech.
  try:
    while True:
      print('Say: ', end='')
      text_to_say = input()
      tts.say(text_to_say, True)
  except KeyboardInterrupt:
    pass

  tts.stop()  # Stop the TTS process.


if __name__ == "__main__":
  main()
