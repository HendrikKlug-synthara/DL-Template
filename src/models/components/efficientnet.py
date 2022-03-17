import torch
from torch import nn
from torchvision.transforms import transforms


class EfficientNet(nn.Module):
    def __init__(
        self,
        output_size: int = 10,
    ):
        super().__init__()
        self.effnet_b0 = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained=True
        )

        num_ftrs = self.effnet_b0.classifier.fc.in_features
        self.effnet_b0.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, output_size),
        )

    def forward(self, x):
        return self.effnet_b0(x)

    @staticmethod
    def get_transform(random_horizontal_flip: bool):
        tfs = [transforms.RandomHorizontalFlip()] if random_horizontal_flip else []
        return {
            "train": transforms.Compose(
                [
                    *tfs,
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                ]
            ),
        }


if __name__ == "__main__":
    model = EfficientNet().train(False)
    x = torch.randn(1, 3, 28, 28)
    model(x)
