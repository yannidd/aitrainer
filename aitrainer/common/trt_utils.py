# *************************************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************************************
"""
The following code has been adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tensorrt/trt_utils.py
"""
import tensorrt as trt
import pycuda.driver as cuda


def is_dimension_dynamic(dim) -> bool:
  """Check if the dimension is dynamic

  Args:
      dim: The dimension.

  Returns:
      bool: 
  """
  return dim is None or dim <= 0


def is_shape_dynamic(shape: trt.Dims) -> bool:
  """Check if the shape is dynamic.

  Args:
      shape (trt.Dims): The shape.

  Returns:
      [bool]: 
  """
  return any([is_dimension_dynamic(dim) for dim in shape])


def run_trt_engine(context: trt.IExecutionContext, engine: trt.ICudaEngine, h_tensors: dict):
  """Run a TRT model.

  The model output is written in place inside the tensors provided in h_tensors['outputs'].

  Args:
      context (trt.IExecutionContext): 
      engine (trt.ICudaEngine): 
      h_tensors (dict): A dictionary with keys "inputs" and "outputs" and values which are another 
      dictionaries with tensor names as keys and numpy.ndarrays as values.
  """
  # Allocate GPU memory.
  d_tensors = {}
  d_tensors['inputs'] = {k: cuda.mem_alloc(v.nbytes) for k, v in h_tensors['inputs'].items()}
  d_tensors['outputs'] = {k: cuda.mem_alloc(v.nbytes) for k, v in h_tensors['outputs'].items()}

  # Copy input buffers to GPU.
  for h_tensor, d_tensor in zip(h_tensors['inputs'].values(), d_tensors['inputs'].values()):
    cuda.memcpy_htod(d_tensor, h_tensor)

  # Initialise bindings list.
  bindings = [None] * engine.num_bindings

  # Populate bindings list.
  for (name, h_tensor), (_, d_tensor) in zip(h_tensors['inputs'].items(),
                                             d_tensors['inputs'].items()):
    idx = engine.get_binding_index(name)
    bindings[idx] = int(d_tensor)
    if engine.is_shape_binding(idx) and is_shape_dynamic(context.get_shape(idx)):
      context.set_shape_input(idx, h_tensor)
    elif is_shape_dynamic(engine.get_binding_shape(idx)):
      context.set_binding_shape(idx, h_tensor.shape)

  for name, d_tensor in d_tensors['outputs'].items():
    idx = engine.get_binding_index(name)
    bindings[idx] = int(d_tensor)

  # Run engine.
  context.execute_v2(bindings=bindings)

  # Copy output buffers to CPU.
  for h_tensor, d_tensor in zip(h_tensors['outputs'].values(), d_tensors['outputs'].values()):
    cuda.memcpy_dtoh(h_tensor, d_tensor)


def load_engine(engine_filepath: str, trt_logger: trt.Logger) -> trt.ICudaEngine:
  """Load a TRT model from path.

  Args:
      engine_filepath (str): The path to the model.
      trt_logger (trt.Logger): An instance of a TRT logger.

  Returns:
      trt.ICudaEngine: The model engine.
  """
  with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
  return engine


def engine_info(engine_filepath: str):
  """Print TRT engine info.

  Args:
      engine_filepath (str): Path to the engine.
  """

  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  engine = load_engine(engine_filepath, TRT_LOGGER)

  binding_template = r"""
{btype} {{
  name: "{bname}"
  data_type: {dtype}
  dims: {dims}
}}"""
  type_mapping = {
      "DataType.HALF": "TYPE_FP16",
      "DataType.FLOAT": "TYPE_FP32",
      "DataType.INT32": "TYPE_INT32",
      "DataType.BOOL": "TYPE_BOOL"
  }

  print("engine name", engine.name)
  print("has_implicit_batch_dimension", engine.has_implicit_batch_dimension)
  start_dim = 0 if engine.has_implicit_batch_dimension else 1
  print("num_optimization_profiles", engine.num_optimization_profiles)
  print("max_batch_size:", engine.max_batch_size)
  print("device_memory_size:", engine.device_memory_size)
  print("max_workspace_size:", engine.max_workspace_size)
  print("num_layers:", engine.num_layers)

  for i in range(engine.num_bindings):
    btype = "input" if engine.binding_is_input(i) else "output"
    bname = engine.get_binding_name(i)
    dtype = engine.get_binding_dtype(i)
    bdims = engine.get_binding_shape(i)
    config_values = {
        "btype": btype,
        "bname": bname,
        "dtype": type_mapping[str(dtype)],
        "dims": list(bdims[start_dim:])
    }
    final_binding_str = binding_template.format_map(config_values)
    print(final_binding_str)


def build_engine(model_file: str,
                 shapes: dict,
                 max_ws: int = 512 * 1024 * 1024,
                 fp16: bool = False) -> trt.ICudaEngine:
  """Build TRT engine from an ONNX file.

  Args:
      model_file (str): Path to the ONNX model.
      shapes (dict): Shapes of inputs.
      max_ws (int, optional): Max workspace size. Defaults to 512*1024*1024.
      fp16 (bool, optional): FP16 inference. Defaults to False.

  Returns:
      trt.ICudaEngine: The resulting engine.
  """
  TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(TRT_LOGGER)
  builder.fp16_mode = fp16

  config = builder.create_builder_config()
  config.max_workspace_size = max_ws
  if fp16:
    config.flags |= 1 << int(trt.BuilderFlag.FP16)
  profile = builder.create_optimization_profile()
  for s in shapes:
    profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
  config.add_optimization_profile(profile)
  explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
  network = builder.create_network(explicit_batch)

  with trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open(model_file, 'rb') as model:
      parsed = parser.parse(model.read())
      for i in range(parser.num_errors):
        print("TensorRT ONNX parser error:", parser.get_error(i))
      engine = builder.build_engine(network, config=config)

      return engine