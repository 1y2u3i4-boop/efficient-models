import torch.nn as nn


def conv_relu(c_in, c_out, k, stride=1):
    return [
        nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        nn.ReLU(inplace=True),
    ]


def make_model(num_classes=100):
    return nn.Sequential(
        *conv_relu(3, 32, 7, stride=2),
        nn.MaxPool2d(3, stride=2, padding=1),
        *conv_relu(32, 64, 5),
        *conv_relu(64, 128, 3, stride=2),
        *conv_relu(128, 256, 1),
        *conv_relu(256, 256, 3, stride=2),
        *conv_relu(256, 512, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )


if __name__ == "__main__":
    import torch
    m = make_model().eval()
    print(m)
    print("params:", sum(p.numel() for p in m.parameters()))
    with torch.inference_mode():
        print(m(torch.randn(2, 3, 224, 224)).shape)
