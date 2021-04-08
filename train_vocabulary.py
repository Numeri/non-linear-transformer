#!/bin/python

from jax import random
import sentencepiece as spm

import dataloader
from hyperparameters import Hyperparameters, hypers

def main():
    key = random.PRNGKey(2021)
    iterator = dataloader.interlaced(hypers, key, 'train', max_length=hypers.max_vocab_sentences)

    model_prefix = f'{hypers.model_folder}/{hypers.vocabulary_prefix}'

    spm.SentencePieceTrainer.train(sentence_iterator=iterator,
            model_prefix=model_prefix,
            vocab_size=hypers.vocabulary_size,
            user_defined_symbols=hypers.extra_tokens
            )

if __name__ == '__main__':
    main()

