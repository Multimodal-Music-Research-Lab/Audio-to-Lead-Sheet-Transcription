from torch import nn


class AudioEncoderBase(nn.Module):
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
