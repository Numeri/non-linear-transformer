#!/bin/python -i
import jax
import numpy
import tensorflow_datasets as tfds

from hyperparameters import Hyperparameters, hypers

def load_datasets(hypers : Hyperparameters):
    builder = tfds.builder(hypers.tensorflow_dataset)

    train      = builder.as_dataset(split='train')
    test       = builder.as_dataset(split='test')
    validation = builder.as_dataset(split='validation')

    return train, test, validation

def sentencepiece_iterator(tf_dataset, language_pair) -> str:
    dataset_iter = iter(tf_dataset)
    for pair in dataset_iter:
        yield pair[language_pair[0]].decode('utf8')
        yield pair[language_pair[1]].decode('utf8')

def tf_to_text(tf_dataset, source_filename, target_filename):
    dataset_iter = iter(tfds.as_numpy(tf_dataset))

    with open(source_filename, 'w', buffering=2**18) as source_outfile, open(target_filename, 'w', buffering=2**18) as target_outfile:
        for pair in dataset_iter:
            source = pair['de'].decode('utf8')
            target = pair['en'].decode('utf8')

            source_outfile.write(source)
            source_outfile.write('\n')
            target_outfile.write(target)
            target_outfile.write('\n')

def main():
    train, test, validation = load_datasets(hypers)
    datasets = {'train':      train,
                'test':       test,
                'validation': validation,
                }

    for dataset_name in datasets.keys():
        print(dataset_name)
        tf_to_text(datasets[dataset_name], f'{hypers.dataset_prefix}/{dataset_name}.{hypers.language_pair[0]}', f'{hypers.dataset_prefix}/{dataset_name}.{hypers.language_pair[1]}')

if __name__ == 'main':
    main()
