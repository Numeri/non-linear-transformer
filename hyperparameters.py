from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # Training parameters
    batch_size : int
    learning_rate : float
    epochs : int
    checkpoint_every : int
    log_every : int
    deterministic : bool
    training_attn_dropout : float

    # Model parameters
    d_model : int
    seq_length : int
    vocabulary_size : int
    extra_tokens : List[str]
    num_heads : int
    num_encoders : int
    num_decoders : int
    beam_width : int

    # Dataset parameters
    language_pair : Tuple[str]
    tensorflow_dataset : str
    dataset_prefix : str
    model_folder : str
    model_name : str
    log_file : str
    vocabulary_prefix : str
    max_vocab_sentences : int

hypers = Hyperparameters(batch_size            = 10,
                         learning_rate         = 0.01,
                         epochs                = 1,
                         checkpoint_every      = 1000,
                         log_every             = 1,
                         deterministic         = False,
                         training_attn_dropout = 0.05,

                         d_model               = 512 // 2,
                         seq_length            = 15,
                         vocabulary_size       = 20000,
                         extra_tokens          = [],
                         num_heads             = 8,
                         num_encoders          = 6,
                         num_decoders          = 6,
                         beam_width            = 5,

                         language_pair         = ('de', 'en'),
                         tensorflow_dataset    = 'wmt14_translate/de-en',
                         dataset_prefix        = 'corpora/wmt14_translate/de-en',
                         model_folder          = 'models',
                         model_name            = 'vanilla_transformer',
                         log_file              = 'log.txt',
                         vocabulary_prefix     = 'sentencepiece',
                         max_vocab_sentences   = 1_000_000,
                         )

