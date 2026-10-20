"""Wan 2.1 I2V-14B (480p) facts that TrajLoc depends on, in one place."""

WAN_CONFIG_NAME = "i2v-14B"  # its entry in Wan2.1's ``WAN_CONFIGS``
TEXT_TOKEN_STD = 0.0734  # per-token std of its UMT5-XXL prompt embeddings
IMAGE_TOKEN_COUNT = 257  # CLIP image tokens ahead of the text in cross-attention
NUM_BLOCKS = 40  # DiT blocks
TEXT_SEQ_LENGTH = 512  # UMT5 context length
LATENT_CHANNELS = 16  # VAE latent channels
I2V_MASK_CHANNELS = (
    4  # the first-frame mask channels ahead of the VAE latent in the I2V conditioning
)
