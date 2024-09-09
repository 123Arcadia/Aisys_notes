import torch
import torch.nn as nn

class Model_Net(nn.Module):
    def __init__(self) -> None:
        super(Model_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        data = self.layer1(data)
        return data


if __name__ == '__main__':
    Batch_size = 8
    Channel = 3
    Height = 256
    Width = 256
    input_data = torch.rand(Batch_size, Channel, Height, Width)

    model = Model_Net()

    # 导出为动态输入
    input_name = 'input'
    output_name = 'output'

    torch.onnx.export(model,
                      input_data,
                      "Dynamics_InputNet.onnx",
                      opset_version=11,
                      export_params=True,
                      input_names=[input_name],
                      output_names=[output_name],
                      dynamic_axes={
                          input_name: {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
                          output_name: {0: 'batch_size', 2: 'output_height', 3: 'output_width'}})