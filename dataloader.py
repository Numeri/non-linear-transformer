import jax
import tensorflow_datasets as tfds

from hyperparameters import Hyperparameters

def load_dataset(hypers : Hyperparameters):
    builder = tfds.builder(hypers.dataset)

    train      = tfds.as_numpy(builder.as_dataset(split='train'))
    test       = tfds.as_numpy(builder.as_dataset(split='train'))
    validation = tfds.as_numpy(builder.as_dataset(split='validation'))

    return train, test, validation

def sentencepiece_generator(dataset, language_pair) -> str:
    dataset_iter = iter(dataset)
    for pair in dataset_iter:
        yield pair[language_pair[0]].decode('utf8')
        yield pair[language_pair[1]].decode('utf8')

