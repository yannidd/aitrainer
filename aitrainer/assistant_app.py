import re
from datetime import date
from aitrainer.asr import ASR
from aitrainer.tts import TTS
from aitrainer.utils.console import setup_logging


def main():
  setup_logging()

  patterns = {
      'what is (.*) plus (.*)': add,
      'what day is it': today,
  }

  asr = ASR(['mike'])
  tts = TTS()

  asr.start()
  tts.start()

  try:
    while True:
      if asr.is_available:
        query = asr.get_text()
        for pattern, fun in patterns.items():
          result = re.search(pattern, query)
          if result:
            text_to_say = None
            try:
              text_to_say = fun(*result.groups())
            except:
              pass
            if text_to_say is not None:
              tts.say(text_to_say, True)
            break
  except KeyboardInterrupt:
    pass

  asr.stop()
  tts.stop()


def add(a, b):
  a = text2int(a)
  b = text2int(b)
  return f'{int2text(a)} plus {int2text(b)} is {int2text(a + b)}.'


def today():
  month = date.today().strftime("%B")
  day = int2text(int(date.today().strftime("%d")))
  return f'today is {month} {day}.'


def text2int(textnum, numwords={}):
  """ Convert a word to an integer.
      Copied from https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers.
  """
  if not numwords:
    units = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    scales = ["hundred", "thousand", "million", "billion", "trillion"]

    numwords["and"] = (1, 0)
    for idx, word in enumerate(units):
      numwords[word] = (1, idx)
    for idx, word in enumerate(tens):
      numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):
      numwords[word] = (10**(idx * 3 or 2), 0)

  current = result = 0
  for word in textnum.split():
    if word not in numwords:
      raise Exception("Illegal word: " + word)

    scale, increment = numwords[word]
    current = current * scale + increment
    if scale > 100:
      result += current
      current = 0

  return result + current


def int2text(n):
  """ Convert an integer to a word.
      Copied from https://stackoverflow.com/questions/19504350/how-to-convert-numbers-to-words-without-using-num2word-library.
  """
  num2words = {
      1: 'One',
      2: 'Two',
      3: 'Three',
      4: 'Four',
      5: 'Five',
      6: 'Six',
      7: 'Seven',
      8: 'Eight',
      9: 'Nine',
      10: 'Ten',
      11: 'Eleven',
      12: 'Twelve',
      13: 'Thirteen',
      14: 'Fourteen',
      15: 'Fifteen',
      16: 'Sixteen',
      17: 'Seventeen',
      18: 'Eighteen',
      19: 'Nineteen',
      20: 'Twenty',
      30: 'Thirty',
      40: 'Forty',
      50: 'Fifty',
      60: 'Sixty',
      70: 'Seventy',
      80: 'Eighty',
      90: 'Ninety',
      0: 'Zero'
  }
  try:
    return num2words[n]
  except KeyError:
    try:
      return num2words[n - n % 10] + num2words[n % 10].lower()
    except KeyError:
      raise('Number out of range')


if __name__ == "__main__":
  main()
