"""One-time export of the trained forgery detection CNN to ONNX format
for lightweight, dependency-safe inference via onnxruntime."""
import torch
from model import ForgeryDetectionCNN

model = ForgeryDetectionCNN()
model.load_state_dict(torch.load("checkpoints/forgery_cnn.pt", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)  # matches training input shape

torch.onnx.export(
    model,
    dummy_input,
    "checkpoints/forgery_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)

print("Exported to checkpoints/forgery_cnn.onnx")