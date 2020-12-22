class CameraBase:
  def __init__(self):
    raise NotImplementedError

  def get_frame(self):
    raise NotImplementedError

  def release(self):
    raise NotImplementedError