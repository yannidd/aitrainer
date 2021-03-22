# Personal AI Trainer

## Prerequisites

 CUDA 10.2 and cuDNN 8.0.3
- Install [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 10.2, [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) 8.0.3, and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) 7.1.3.4.

- Install [PyTorch](https://pytorch.org/) with CUDA.

- Create a Python environment (optional, but recommended):

```bash
python3 -m venv .env  # Create an environment.
source .env/bin/activate  # Activate it.
```

- Install Python dependencies:

```bash
pip install -r requirements.txt
```

- Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt):

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --plugins
cd ..
rm -rf torch2trt
```

- Install [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose):
```bash
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
python setup.py install
cd ..
rm -rf trt_pose
```

- Install [ctcdecode](https://github.com/parlance/ctcdecode):

```bash
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..
rm -rf ctcdecode
```

Download the models from TODO and unzip them. Then generate the TensorRT enignes using:

```bash
python -m scripts.onnx2trt
```

## Running Apps:

```bash
python -m aitrainer.asr_app  # ASR App
python -m aitrainer.tts_app  # TTS App
python -m aitrainer.assistant_app  # Assistant App
```