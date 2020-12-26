from aitrainer.asr import ASR
from aitrainer.camera import CameraCv2
from aitrainer.pose import PoseEstimator
from aitrainer.tts import TTS
from aitrainer.utils.console import setup_logging


def main():
  setup_logging()

  asr = ASR(['jarvis', 'jervis', 'gervis'])
  tts = TTS()
  pose = PoseEstimator(CameraCv2)

  asr.start()
  tts.start()
  pose.start()

  tts.say('Application Started!')
  tts.say('Let\'s do a workout.')

  try:
    while True:
      if asr.is_available:
        pass
      #   tts.say(asr.get_text() + '.')
      if pose.keypoints_available:
        pose.get_keypoints()
  except KeyboardInterrupt:
    pass

  asr.stop()
  tts.stop()
  pose.stop()


if __name__ == "__main__":
  main()
