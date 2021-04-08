import jax
from jax import numpy as np, random

from typing import Union
import linecache

import sentencepiece as spm
from hyperparameters import Hyperparameters

def interlaced(hypers : Hyperparameters, key : np.ndarray, dataset_name : str, max_length : int):
    source_filename = f'{hypers.dataset_prefix}/{dataset_name}.{hypers.language_pair[0]}'
    target_filename = f'{hypers.dataset_prefix}/{dataset_name}.{hypers.language_pair[1]}'

    _ = linecache.getline(source_filename, 0)
    _ = linecache.getline(target_filename, 0)

    line_count = len(linecache.cache[source_filename][2])

    indices = np.arange(line_count)
    indices = random.permutation(key, indices)

    for i in indices[:max_length // 2]:
        yield linecache.getline(source_filename, i)
        yield linecache.getline(target_filename, i)


def get_batches(hypers : Hyperparameters, key : np.ndarray, dataset_name : str):
    sp = spm.SentencePieceProcessor(model_file=f'{hypers.model_folder}/{hypers.vocabulary_prefix}.model')

    source_filename = f'{hypers.dataset_prefix}/{dataset_name}.{hypers.language_pair[0]}'
    target_filename = f'{hypers.dataset_prefix}/{dataset_name}.{hypers.language_pair[1]}'

    _ = linecache.getline(source_filename, 0)
    _ = linecache.getline(target_filename, 0)

    line_count = len(linecache.cache[source_filename][2])

    indices = np.arange(line_count)
    indices = random.permutation(key, indices)

    for batch_start in range(0, line_count, hypers.batch_size):
        batch_end = batch_start + hypers.batch_size
        batch_end = np.minimum(batch_end, line_count)

        sources = [process_sentence(hypers, sp, linecache.getline(source_filename, indices[i]))
                    for i in range(batch_start, batch_end)]

        targets = [process_sentence(hypers, sp, linecache.getline(target_filename, indices[i]))
                    for i in range(batch_start, batch_end)]

        sources = np.stack(sources)
        targets = np.stack(targets)

        yield sources, targets


def process_sentence(hypers : Hyperparameters, sp : spm.SentencePieceProcessor, sentence : Union[str, bytes]):
    sentence = sp.encode(sentence)
    sentence = np.array(sentence, dtype='int32')
    padding = np.maximum(0, hypers.seq_length - sentence.shape[0])
    sentence = np.pad(sentence, [0, padding])
    sentence = sentence[:hypers.seq_length]

    return sentence

