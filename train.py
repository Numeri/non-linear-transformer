import flax
import jax

from flax import linen as nn
from jax import numpy as np, random

import time

from model import Transformer
from decode import max_decode_logits
from dataloader import get_batches
from hyperparameters import Hyperparameters, hypers

jax.profiler.start_trace('/tmp/tensorboard')

def log(s : str):
    with open(hypers.log_file, 'a') as outfile:
        outfile.write(s)
        outfile.write('\n')

key = random.PRNGKey(11)

last_time = time.perf_counter()

# Initialize model
model = Transformer(hypers)
dummy = np.zeros((hypers.batch_size, hypers.seq_length), dtype=int)

log(f'Model initialized ({time.perf_counter() - last_time:7.3f} seconds)')
last_time = time.perf_counter()

# Initialize parameters
key, key_, rng = random.split(key, 3)
params = model.init({'dropout': rng, 'params': key_}, dummy, dummy)

log(f'Parameters initialized ({time.perf_counter() - last_time:7.3f} seconds)')
last_time = time.perf_counter()

def make_batched_loss(key, source_batch, target_batch):
    def loss(params):
        def cross_entropy_sentence(source, target):
            """Cross entropy loss for a single source-target sentence pair"""
            logits = max_decode_logits(hypers, key, model, params, source)
            target_logits = jax.nn.one_hot(target, hypers.vocabulary_size, dtype='float32')
            weights = np.where(logits.sum(axis=-1) > 0, 1, 0)

            cross_entropies = -weights*np.sum(target_logits * np.log(logits), axis=-1)

            return np.mean(cross_entropies, axis=-1)

        sentence_losses = jax.vmap(cross_entropy_sentence)(source_batch, target_batch)
        return np.mean(sentence_losses)
    return jax.jit(loss)

# Set up optimizer
optimizer = flax.optim.Adam(hypers.learning_rate).create(params)

log(f'Starting training ({time.perf_counter() - last_time:7.3f} seconds)')
last_time = time.perf_counter()

report_times = np.array([], dtype='float32')

# Training loop
for epoch in range(hypers.epochs):
    # Get training batches
    key, key_ = random.split(key)
    training_batches = get_batches(hypers, key_, 'train')
    
    for batch_num, (source_batch, target_batch) in enumerate(training_batches):
        key, key_ = random.split(key)
        loss = make_batched_loss(key_, source_batch, target_batch)
        loss_val_grad = jax.value_and_grad(loss)

        loss_val, grad = loss_val_grad(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)

        if batch_num % hypers.checkpoint_every == 0:
            serial_params = flax.serialization.to_bytes(optimizer.target)
            with open(f'{hypers.model_folder}/{hypers.model_name}_{batch_num}.params', 'wb') as outfile:
                outfile.write(serial_params)

        if batch_num % hypers.log_every == 0:
            time_since_report = time.perf_counter() - last_time
            report_times = np.append(report_times, time_since_report)
            per_batch = report_times.mean() / hypers.log_every
            batches_per_epoch = 9_000_000 / hypers.batch_size
            estimated_time = per_batch * ((batches_per_epoch - batch_num) + batches_per_epoch*(hypers.epochs - epoch - 1))
            log(f'Batch: {batch_num}\tloss: {loss_val}\tmean batch time: {per_batch}\testimated time remaining: {estimated_time}')
            last_time = time.perf_counter()
            
            jax.profiler.stop_trace()

