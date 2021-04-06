import flax
import jax

from flax import linen as nn
from jax import numpy as np, random

from hyperparameters import Hyperparameters
from model import Transformer


np.set_printoptions(edgeitems=30, linewidth=190)
hypers = Hyperparameters(batch_size            = 10,
                         learning_rate         = 0.01,
                         epochs                = 1000,
                         deterministic         = False,
                         training_attn_dropout = 0.05,

                         d_model               = 512 // 2,
                         seq_length            = 256 // 2,
                         vocabulary_size       = 1000 // 2,
                         num_heads             = 8,
                         num_encoders          = 6,
                         num_decoders          = 6,
                         )

key = random.PRNGKey(10)
model = Transformer(hypers)
dummy = np.zeros((hypers.batch_size, hypers.seq_length), dtype=int)

key, key_, rng = random.split(key, 3)
params = model.init({'dropout': rng, 'params': key_}, dummy, dummy)

