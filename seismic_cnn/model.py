import torch
import torch.nn as nn


class SeismicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # shape is now (batch, 32, 3000)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # shape is now (batch, 64,1500)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(5),  # shape is now (batch,128, 300)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # shape is now (batch, 128,1)

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Removes the trailing 1 dimension. (batch, 128, 1) → (batch, 128). Linear layers don't accept 3D input, so you need to flatten first.
            nn.Linear(
                128, 64
            ),  # First FC layer: Combines the 128 filter signals into 64 intermediate features. Learns which combinations of filters are meaningful.
            nn.ReLU(),  # Keeps only positive combinations, throws away negative ones. Adds non-linearity so the model can learn complex decision boundaries, not just straight lines.
            nn.Dropout(
                0.4
            ),  # During training, randomly zeroes 40% of the 64 neurons on each forward pass. Forces the model not to rely too heavily on any single neuron — reduces overfitting.
            nn.Linear(
                64, 1
            ),  # Takes the 64 intermediate features and compresses to a single score. This is the raw "earthquake-ness" value before converting to probability.
            nn.Sigmoid(),  # Converts any number to a value between 0 and 1: very negative → ~0.0  (noise), near zero     → ~0.5  (uncertain)
            # very positive → ~1.0  (earthquake)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
