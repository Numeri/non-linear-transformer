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
    print(s)

def learning_rate_fn(step_num):
    return (hypers.d_model**-0.5) * np.minimum(step_num**-0.5, step_num*hypers.warmup_steps**-1.5)

@jax.jit
def train_step(key, optimizer, source_batch, target_batch, step_num):
    def loss(params):
        logits = model.apply(params, source_batch, target_batch, rngs={'dropout': key})
        target_logits = jax.nn.one_hot(target_batch, hypers.vocabulary_size, dtype='float32')
        weights = np.where(target_batch > 0, 1, 0)

        cross_entropies = -weights*np.sum(target_logits * np.log(logits), axis=-1)
        losses = np.sum(cross_entropies, axis=-1)

        return np.mean(losses)

    loss_val, grad = jax.value_and_grad(loss)(optimizer.target)
    lr = learning_rate_fn(step_num)
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

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

# Set up optimizer
optimizer = flax.optim.Adam(
        learning_rate=1,
        beta1=hypers.adam_beta_1,
        beta2=hypers.adam_beta_2,
        eps=hypers.adam_epsilon
        ).create(params)
start_batch = 0

if hypers.restore_checkpoint is not None:
    with open(f'{hypers.model_folder}/{hypers.model_name}_{hypers.restore_checkpoint}.params', 'rb') as infile:
        optimizer = flax.serialization.from_bytes(optimizer, infile.read())
        optimizer = jax.tree_map(np.asarray, optimizer)
    start_batch = hypers.restore_checkpoint

log(f'Starting training ({time.perf_counter() - last_time:7.3f} seconds)')
last_time = time.perf_counter()

report_times = np.array([], dtype='float32')

step_num = start_batch

# Training loop
for epoch in range(hypers.epochs):
    # Get training batches
    key, key_ = random.split(key)
    training_batches = get_batches(hypers, key_, 'train', start_batch)
    
    for batch_num, (source_batch, target_batch) in enumerate(training_batches):
        # batch_num is 0..n, but we might start partway through after restoring a checkpoint
        batch_num += start_batch
        step_num += 1

        key, key_ = random.split(key)
        optimizer, loss_val = train_step(
                key_,
                optimizer,
                source_batch,
                target_batch,
                step_num, 
                )

        if batch_num % hypers.checkpoint_every == 0:
            bytes_optimizer = flax.serialization.to_bytes(optimizer)
            with open(f'{hypers.model_folder}/{hypers.model_name}_{batch_num}.params', 'wb') as outfile:
                outfile.write(bytes_optimizer)
            
            batch_size = hypers.batch_size
            hypers.batch_size = 10
            validation_batches = get_batches(hypers, random.PRNGKey(1), 'validation')
            val_src_batch, val_trg_batch = next(validation_batches)
            eval_step(optimizer, val_src_batch, val_trg_batch, batch_num)
            hypers.batch_size = batch_size

        if batch_num % hypers.log_every == 0:
            time_since_report = time.perf_counter() - last_time

            report_times = np.append(report_times, time_since_report)

            if len(report_times) > 1:
                per_sentence = report_times[1:].mean() / (hypers.log_every * hypers.batch_size)
                sentences_per_epoch = 9_000_000
                remaining_sentences = sentences_per_epoch - batch_num * hypers.batch_size + sentences_per_epoch * (hypers.epochs - epoch - 1)
                estimated_time = per_sentence * remaining_sentences
            else:
                per_sentence = '--'
                estimated_time = '--'


            log(f'Batch: {batch_num}\tloss: {loss_val}\tbatch time: {time_since_report}\tmean sentence time: {per_sentence}\testimated time remaining: {estimated_time}')

            last_time = time.perf_counter()

    start_batch = 0

