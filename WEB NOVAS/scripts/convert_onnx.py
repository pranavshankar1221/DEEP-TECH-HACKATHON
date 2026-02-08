import torch
import torch.nn as nn
import torch.nn.functional as F

# Load checkpoint
checkpoint = torch.load("best_cnn_phase1.pth", map_location="cpu")
class_names = checkpoint['class_names']

class MyCustomCNN(nn.Module):
    def __init__(self, num_classes=len(class_names)):
        super(MyCustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyCustomCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 1, 64, 64)
onnx_file = "wafer_model.onnx"
torch.onnx.export(
    model, dummy_input, onnx_file,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
print(f"Model exported to ONNX: {onnx_file}")
