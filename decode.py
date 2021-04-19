import jax
from jax import numpy as np

from typing import Any, Union
import sentencepiece as spm

import dataloader
from model import Transformer
from hyperparameters import Hyperparameters

JaxArray = jax.interpreters.xla._DeviceArray

def max_decode(hypers : Hyperparameters, model : Transformer, params : Any,  source : Union[JaxArray, str]):
    sp = spm.SentencePieceProcessor(model_file=f'{hypers.model_folder}/{hypers.vocabulary_prefix}.model')
    eos_id = sp.piece_to_id('</s>')

    if type(source) is str:
        source = np.array(dataloader.process_sentence(hypers, sp, source))

    source = np.expand_dims(source, 0)

    decoded_seq = np.zeros((1, hypers.seq_length), dtype='int32')
    position = 0
    log_score = 0.0

    model.hypers.deterministic = True

    while not np.isin(decoded_seq, eos_id).any() and position < hypers.seq_length - 1:
        logits = model.apply(params, source, decoded_seq)
        max_next_word = np.argmax(logits[0, position])
        log_score = log_score + np.log(logits[:, position, max_next_word])
        decoded_seq = jax.ops.index_update(decoded_seq, (0,position), max_next_word)
        position += 1

    decoded_str = sp.decode(decoded_seq[0].tolist())

    return decoded_str, log_score

def max_decode_logits(hypers : Hyperparameters, key : np.ndarray, model : Transformer, params : Any,  source : Union[JaxArray, str]):
    sp = spm.SentencePieceProcessor(model_file=f'{hypers.model_folder}/{hypers.vocabulary_prefix}.model')
    eos_id = sp.piece_to_id('</s>')

    if type(source) is str:
        source = np.array(dataloader.process_sentence(hypers, sp, source))

    source = np.expand_dims(source, 0)

    decoded_seq = np.zeros((1, hypers.seq_length), dtype='int32')

    def update(decoded_seq, position):
        logits = model.apply(params, source, decoded_seq, rngs={'dropout': key})
        max_next_word = np.argmax(logits[0, position])
        decoded_seq = jax.ops.index_update(decoded_seq, (0,position), max_next_word)

        return decoded_seq, logits[0, position]

    decoded_seq, all_logits = jax.lax.scan(update, decoded_seq, np.arange(hypers.seq_length))

    decoded_seq = sp.decode(decoded_seq[0].tolist())

    return decoded_seq, all_logits

def beam_search(hypers : Hyperparameters, model : Transformer, params : Any,  source : Union[JaxArray, str]):
    sp = spm.SentencePieceProcessor(model_file=f'{hypers.model_folder}/{hypers.vocabulary_prefix}.model')
    eos_id = sp.piece_to_id('</s>')

    def top_next(n : int, position : int, source : JaxArray, decoded_seqs : JaxArray, log_scores : JaxArray):
        breakpoint()
        logits = model.apply(params, source, decoded_seqs)
        top_words = np.argsort(-logits[:, position])[:, :n]
        top_log_scores = np.log(np.take_along_axis(logits[:, position, :], top_words[:, :n], axis=-1))

        # Create list of all combinations, so we can sort by score
        all_options = [(seq_index,
                        int(top_words[seq_index, word_index]), 
                        float(top_log_scores[seq_index, word_index] + log_scores[seq_index])
                        )
                       for seq_index in range(decoded_seqs.shape[0])
                       for word_index in range(n)
                      ]

        # Sort all options largest to smallest by score
        top_options = sorted(all_options, key=lambda x : x[2], reverse=True)[:n]

        # Invert the list of n 3-tuples into a list of 3 n-tuples
        # then extract the relevant values
        seq_indices, top_words, top_log_scores = list(zip(*top_options))

        new_sequences = decoded_seqs[seq_indices, :]

        for i, word in enumerate(top_words):
            new_sequences = jax.ops.index_update(new_sequences, (i, position), word)

        return new_sequences, np.array(top_log_scores)

    def is_finished(decoded_seq : JaxArray) -> bool:
        return np.isin(decoded_seq, eos_id).any()

    if type(source) is str:
        source = np.array(dataloader.process_sentence(hypers, sp, source))

    source = np.expand_dims(source, 0)

    top_seqs       = np.zeros((1, hypers.seq_length), dtype='int32')
    top_log_scores = np.zeros((hypers.beam_width,))
    finished       = np.zeros((hypers.beam_width,), dtype='bool')
    position       = 0

    top_seqs, top_log_scores = top_next(hypers.beam_width, position, source, top_seqs, top_log_scores)
    position += 1

    source = np.tile(source, (hypers.beam_width, 1))

    while not finished.all() and position < hypers.seq_length - 1:
        top_seqs, top_log_scores = top_next(hypers.beam_width, position, source[~finished], top_seqs[~finished], top_log_scores[~finished])
        position += 1

        finished = jax.vmap(is_finished)(top_seqs)

    decoded_str = sp.decode(top_seqs[0].tolist())

    return decoded_str, top_log_scores[0]

