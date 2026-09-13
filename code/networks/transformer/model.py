import torch
from lightning.pytorch import LightningModule

from my_utils.consts import PREPROCESSED_MUQ_ENCODER
from networks.transformer.decoder import Decoder
from networks.transformer.muq_encoder import MuqEncoderPreprocessed


class A2STransformer(LightningModule):
    def __init__(
        self,
        max_seq_len,
        max_audio_len,
        w2i,
        i2w,
        max_encoder_output_length,
        ytest_i2w=None,
        attn_window=-1,
        teacher_forcing_prob=0.5,
        encoder=PREPROCESSED_MUQ_ENCODER,
        lr=1e-4,
        weight_decay=0.0,
        ff_dim_multiplier=1,
        label_smoothing=0.0,
        use_pre_norm=True,
        lyrics_bpe: dict | None = None,
        model_arch: str = "flat",
        decoder_dim: int | None = None,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.label_smoothing = label_smoothing
        self.w2i = w2i
        self.i2w = i2w
        self.padding_idx = w2i["<PAD>"]
        # Model
        self.max_seq_len = max_seq_len
        self.max_flattened_encoder_output_length = max_encoder_output_length

        self.encoder = self._build_encoder(max_encoder_output_length, max_audio_len)

        embedding_dim = decoder_dim or self.encoder.get_output_dim()
        self.decoder_dim = embedding_dim
        self.decoder = Decoder(
            output_size=len(self.w2i),
            max_seq_len=self.max_seq_len,
            num_embeddings=len(self.w2i),
            padding_idx=self.padding_idx,
            attn_window=attn_window,
            embedding_dim=embedding_dim,
            ff_dim=embedding_dim * ff_dim_multiplier,
            use_pre_norm=use_pre_norm,
        )

    @property
    def lyrics_bpe_tokeniser(self):
        if hasattr(self, "_lyrics_bpe_tokeniser"):
            return self._lyrics_bpe_tokeniser
        metadata = self.hparams.get("lyrics_bpe")
        if metadata is None:
            return None
        from my_utils.lyrics_bpe_tokeniser import LyricsBPETokeniser

        self._lyrics_bpe_tokeniser = LyricsBPETokeniser(metadata["model_name"])
        return self._lyrics_bpe_tokeniser

    def _build_encoder(self, max_encoder_output_length, max_audio_len):
        return MuqEncoderPreprocessed(
            max_encoder_output_length=max_encoder_output_length,
            max_audio_len=max_audio_len,
        )

    def forward(self, x, xl, y_in):
        x = self.encoder(x=x)

        # Decoder
        y_out_hat = self.decoder(tgt=y_in, memory=x, memory_len=xl)

        return y_out_hat


def model_class_for_encoder(encoder: str):
    return A2STransformer
