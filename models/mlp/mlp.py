import torch
import torch.nn as nn
import torch.onnx
import os

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Example: input is a batch of 1, 10 features
    model = SimpleMLP(input_dim=10, hidden_dim=32, output_dim=2)
    model.eval()
    dummy_input = torch.randn(1, 10)
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print("Exported simple_mlp.onnx")
