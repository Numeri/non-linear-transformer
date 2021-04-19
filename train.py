import flax
import jax

from flax import linen as nn
from jax import numpy as np, random
import sentencepiece as spm

import time

from model import Transformer
from decode import max_decode_logits, beam_search, max_decode
from dataloader import get_batches
from hyperparameters import Hyperparameters, hypers

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

@jax.jit
def train_step(key, optimizer, source_batch, target_batch):
    def loss(params):
        logits = model.apply(params, source_batch, target_batch, rngs={'dropout': key})
        target_logits = jax.nn.one_hot(target_batch, hypers.vocabulary_size, dtype='float32')
        weights = np.where(target_batch > 0, 1, 0)

        cross_entropies = -weights*np.sum(target_logits * np.log(logits), axis=-1)

        return np.mean(cross_entropies)

    loss_val, grad = jax.value_and_grad(loss)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)

    return optimizer, loss_val

sp = spm.SentencePieceProcessor(model_file=f'{hypers.model_folder}/{hypers.vocabulary_prefix}.model')
def eval_step(optimizer, source_batch, target_batch, batch_num):
    model.hypers.deterministic = True
    with open(f'eval/{batch_num}.txt', 'w') as outfile:
        for i in range(source_batch.shape[0]):
            decoded_seq, _ = max_decode_logits(hypers, key, model, optimizer.target, source_batch[i])
            outfile.write(sp.decode(source_batch[i].tolist()))
            outfile.write('\n')
            outfile.write(sp.decode(target_batch[i].tolist()))
            outfile.write('\n')
            outfile.write(decoded_seq)
            outfile.write('\n\n')
    model.hypers.deterministic = False

# Set up optimizer
optimizer = flax.optim.Adam(hypers.learning_rate).create(params)
start_batch = 0

if hypers.restore_checkpoint is not None:
    with open(f'{hypers.model_folder}/{hypers.model_name}_{hypers.restore_checkpoint}.params', 'rb') as infile:
        target = flax.serialization.from_bytes(optimizer.target, infile.read())
        optimizer = flax.optim.Adam(hypers.learning_rate).create(params)
    start_batch = hypers.restore_checkpoint

log(f'Starting training ({time.perf_counter() - last_time:7.3f} seconds)')
last_time = time.perf_counter()

report_times = np.array([], dtype='float32')

# Training loop
for epoch in range(hypers.epochs):
    # Get training batches
    key, key_ = random.split(key)
    training_batches = get_batches(hypers, key_, 'train', start_batch)
    
    for batch_num, (source_batch, target_batch) in enumerate(training_batches):
        # batch_num is 0..n, but we might start partway through after restoring a checkpoint
        batch_num += start_batch

        key, key_ = random.split(key)
        optimizer, loss_val = train_step(key_, optimizer, source_batch, target_batch)

        if batch_num % hypers.checkpoint_every == 0:
            bytes_optimizer = flax.serialization.to_bytes(optimizer)
            with open(f'{hypers.model_folder}/{hypers.model_name}_{batch_num}.params', 'wb') as outfile:
                outfile.write(bytes_optimizer)
            
            validation_batches = get_batches(hypers, random.PRNGKey(1), 'validation')
            val_src_batch, val_trg_batch = next(validation_batches)
            eval_step(optimizer, val_src_batch, val_trg_batch, batch_num)

        if batch_num % hypers.log_every == 0:
            time_since_report = time.perf_counter() - last_time

            report_times = np.append(report_times, time_since_report)

            if len(report_times) > 1:
                per_batch = report_times[1:].mean() / hypers.log_every
                batches_per_epoch = 9_000_000 / hypers.batch_size
                estimated_time = per_batch * ((batches_per_epoch - batch_num) + batches_per_epoch*(hypers.epochs - epoch - 1))
            else:
                per_batch = '--'
                batches_per_epoch = '--'
                estimated_time = '--'


            log(f'Batch: {batch_num}\tloss: {loss_val}\tbatch time: {time_since_report}\tmean batch time: {per_batch}\testimated time remaining: {estimated_time}')

            last_time = time.perf_counter()

