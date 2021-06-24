from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # Training parameters
    batch_size : int
    warmup_steps : float
    adam_beta_1 : float
    adam_beta_2 : float
    adam_epsilon : float
    epochs : int
    checkpoint_every : int
    log_every : int
    deterministic : bool
    dropout_rate : float
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
    restore_checkpoint : Optional[int]
    log_file : str
    vocabulary_prefix : str
    max_vocab_sentences : int

hypers = Hyperparameters(batch_size            = 100,
                         warmup_steps          = 3000,
                         adam_beta_1           = 0.9,
                         adam_beta_2           = 0.98,
                         adam_epsilon          = 10e-9,
                         epochs                = 1,
                         checkpoint_every      = 500,
                         log_every             = 20,
                         deterministic         = False,
                         dropout_rate          = 0.10,
                         training_attn_dropout = 0.10,

                         d_model               = 512,
                         seq_length            = 50,
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
                         restore_checkpoint    = None,
                         log_file              = 'log.txt',
                         vocabulary_prefix     = 'sentencepiece',
                         max_vocab_sentences   = 1_000_000,
                         )

