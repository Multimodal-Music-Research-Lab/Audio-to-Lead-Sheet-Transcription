from networks.transformer.decoder import PositionalEncoding1D
from networks.transformer.encoder_modules import AudioEncoderBase
from torch import nn

EMB_LAYER_SIZE = 256


class LinearBridge(nn.Module):
    def __init__(self, muq_dim=1024, model_dim=256):
        super().__init__()
        self.proj = nn.Linear(muq_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x


class MuqEncoderPreprocessed(AudioEncoderBase):
    def __init__(self, max_encoder_output_length, max_audio_len):
        super().__init__()

        self.pe = PositionalEncoding1D(
            emb_dim=EMB_LAYER_SIZE,
            max_len=max_encoder_output_length,
        )

        self.proj_layer = LinearBridge()

    def forward(self, x):
        # Dataset samples are [time, features], while inference passes a
        # single sample with a leading batch dimension.
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(
                "Preprocessed MuQ input must have shape [batch, time, features] "
                "or [time, features]"
            )
        x = self.proj_layer(x)
        x = self.pe(x)
        return x

    def get_output_dim(self) -> int:
        return EMB_LAYER_SIZE
