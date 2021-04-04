import flax
import jax

from flax import linen as nn
from jax import numpy as np

import functools

from hyperparameters import Hyperparameters

JaxArray = jax.interpreters.xla._DeviceArray

np.set_printoptions(edgeitems=30, linewidth=190)
hypers = Hyperparameters(batch_size = 1000,
                         learning_rate = 0.01,
                         epochs = 1000,
                         d_model = 512,
                         seq_length = 256,
                         num_heads = 8,
                         )

class PositionalEmbedding(nn.Module):
    """Create sinusoidal positional encodings for each position in the input

    Attributes:
      hypers : Hyperparameters for the model
    """
    hypers : Hyperparameters

    def __call__(self, x):
        def sines(pos : int, i : int) -> float:
            scaled_pos = pos / (10000**(i / self.hypers.d_model))
            return np.sin(scaled_pos)
        def cosines(pos : int, i : int) -> float:
            scaled_pos = pos / (10000**(i / self.hypers.d_model))
            return np.cos(scaled_pos)

        positions = np.arange(self.hypers.seq_length)
        dimensions = np.arange(self.hypers.d_model // 2)

        pos_emb_sines   =   sines(positions[:, np.newaxis], dimensions[np.newaxis, :])
        pos_emb_cosines = cosines(positions[:, np.newaxis], dimensions[np.newaxis, :])

        pos_emb = np.concatenate([pos_emb_sines, pos_emb_cosines], axis=1)
        pos_emb = np.expand_dims(pos_emb, axis=0)

        assert(pos_emb.shape[1:] == x.shape[1:])

        return x + pos_emb


class Embedding(nn.Module):
    """Embed previously tokenized input into vectors of size d_model
    
    Attributes:
      hypers : Hyperparameters for the model
      shared_embedding : An nn.Embed module shared between the input and output embeddings, and the decoding linear transformation
    """
    hypers : Hyperparameters
    shared_embedding : nn.Embed

    @nn.compact
    def __call__(self, x : JaxArray) -> JaxArray:
        x = self.shared_embedding(x)
        x = PositionalEmbedding(hypers=hypers)(x)

        return x

class FeedForward(nn.Module):
    """FeedForward block with two dense networks

    Attributes:
      hypers : Hyperparameters for the model
    """
    hypers : Hyperparameters

    @nn.compact
    def __call__(self, x : JaxArray) -> JaxArray:
        x = nn.Dense(features=hypers.d_model)(x)
        x = nn.Dense(features=hypers.d_model)(x)

        return x

class Encoder(nn.Module):
    """Encoder layer

    Attributes:
      hypers : Hyperparameters for the model
    """
    hypers : Hyperparameters

    @nn.compact
    def __call__(self, x : JaxArray) -> JaxArray:
        multihead_residual = x
        x = nn.attention.MultiHeadDotProductAttention(
                num_heads=hypers.num_heads,
                dropout_rate=hypers.attn_dropout_rate,
                deterministic=hypers.deterministic)(x, x)
        x = x + multihead_residual
        x = nn.LayerNorm()(x)

        feed_forward_residual = x
        x = FeedForward(hypers)(x)
        x = x + feed_forward_residual
        x = nn.LayerNorm()(x)

        return x

