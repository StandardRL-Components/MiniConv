from embedder import process_conv_layers
import torch.nn as nn

# Example usage:
model = nn.Sequential(
    nn.Conv2d(4, 8, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(8, 16, kernel_size=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(16, 4, kernel_size=2),
    nn.ReLU()
)

process_conv_layers(model, "ExampleModel")