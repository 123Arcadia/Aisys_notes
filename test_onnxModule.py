import numpy as np
import onnx
import onnxruntime

input_data1 = np.random.rand(4,3,256,256).astype(np.float32)
input_data2 = np.random.rand(8,3,256,256).astype(np.float32)

Onnx_file = "./Dynamics_InputNet.onnx"
Model = onnx.load(Onnx_file)
onnx.checker.check_model(Model)  # 验证 ONNX 模型是否准确
print('-'*5, 'onnx.helper.printable_graph(Model.graph)', '-'*5)
print(onnx.helper.printable_graph(Model.graph))
# graph torch_jit (
#   %input[FLOAT, batch_sizex3xinput_heightxinput_width]
# ) initializers (
#   %onnx::Conv_22[FLOAT, 64x3x3x3]
#   %onnx::Conv_23[FLOAT, 64]
#   %onnx::Conv_25[FLOAT, 256x64x3x3]
#   %onnx::Conv_26[FLOAT, 256]
# ) {
#   %/layer1/layer1.0/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%input, %onnx::Conv_22, %onnx::Conv_23)
#   %/layer1/layer1.2/Relu_output_0 = Relu(%/layer1/layer1.0/Conv_output_0)
#   %/layer1/layer1.3/Conv_output_0 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]](%/layer1/layer1.2/Relu_output_0, %onnx::Conv_25, %onnx::Conv_26)
#   %output = Relu(%/layer1/layer1.3/Conv_output_0)
#   return %output
# }

# 使用 onnxruntime 进行推理
# 创建推理会话
"""
>>> onnxruntime.get_available_providers()
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
"""
model = onnxruntime.InferenceSession(Onnx_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = model.get_inputs()[0].name  # 获取模型输入的名称
output_name = model.get_outputs()[0].name  # 获取模型输出的名称
# NodeArg(name='input', type='tensor(float)', shape=['batch_size', 3, 'input_height', 'input_width'])
# NodeArg(name='output', type='tensor(float)', shape=['batch_size', 256, 'output_height', 'output_width'])

# 对两组输入数据进行推理
output1 = model.run([output_name], {input_name: input_data1})  # 对第一组输入数据进行推理
output2 = model.run([output_name], {input_name: input_data2})  # 对第二组输入数据进行推理

# 打印输出结果的形状
print('output1.shape: ', np.squeeze(np.array(output1), 0).shape)  # 打印第一组输入数据的输出结果形状
print('output2.shape: ', np.squeeze(np.array(output2), 0).shape)  # 打印第二组输入数据的输出结果形

# output1.shape:  (4, 256, 256, 256)
# output2.shape:  (8, 256, 256, 256)

print(onnxruntime.get_available_providers.__str__())
print('----------------------')
print(model.get_inputs(), type(model.get_inputs()[0]), type(model.get_outputs()[0]))
print(model.get_outputs())
print(model.get_inputs()[0])
# NodeArg(name='input', type='tensor(float)', shape=['batch_size', 3, 'input_height', 'input_width'])
print(model.get_outputs()[0])
# NodeArg(name='output', type='tensor(float)', shape=['batch_size', 256, 'output_height', 'output_width'])

