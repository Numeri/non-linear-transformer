import flax
import jax

from flax import linen as nn
from jax import numpy as np, random

from hyperparameters import Hyperparameters, hypers
from model import Transformer


np.set_printoptions(edgeitems=30, linewidth=190)

key = random.PRNGKey(10)
model = Transformer(hypers)
dummy = np.zeros((hypers.batch_size, hypers.seq_length), dtype=int)

key, key_, rng = random.split(key, 3)
params = model.init({'dropout': rng, 'params': key_}, dummy, dummy)

