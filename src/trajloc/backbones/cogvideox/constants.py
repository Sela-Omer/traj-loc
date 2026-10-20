"""CogVideoX-5B I2V facts that TrajLoc depends on, in one place."""

MODEL_ID = "zai-org/CogVideoX-5b-I2V"
MODEL_REVISION = "a6f0f4858a8395e7429d82493864ce92bf73af11"  # its hub revision
VAE_MODEL_ID = "zai-org/CogVideoX-5b"  # Stage 2 encodes with the T2V model's VAE
VAE_MODEL_REVISION = "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321"
NUM_BLOCKS = 42  # transformer blocks
TEXT_SEQ_LENGTH = 226  # T5 tokens per prompt, padded or truncated
TEXT_TOKEN_STD = 0.15  # per-token std of its T5-XXL prompt embeddings
LATENT_SCALE = 0.7  # Stage 2 stores VAE latents multiplied by this
I2V_IN_CHANNELS = 32  # the I2V transformer's input: noisy latents + image latents, 16 each
