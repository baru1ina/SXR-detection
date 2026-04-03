import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # causal trim
        out = self.relu(out)

        res = x if self.downsample is None else self.downsample(x)

        return out + res


class TCNPredictor(nn.Module):

    def __init__(self, input_dim, num_channels=[32, 32], kernel_size=3):
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation=2 ** i
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], input_dim)

    def forward(self, x):
        x = x.transpose(1, 2)

        out = self.network(x)

        last = out[:, :, -1]

        return self.fc(last)