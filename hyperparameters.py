from typing import Tuple
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # Training parameters
    batch_size : int
    learning_rate : float
    epochs : int
    deterministic : bool
    training_attn_dropout : float

    # Model parameters
    d_model : int
    seq_length : int
    vocabulary_size : int
    extra_tokens : Tuple[str]
    num_heads : int
    num_encoders : int
    num_decoders : int

    # Dataset parameters
    dataset : str
    model_folder : str
    vocabulary_prefix : str

hypers = Hyperparameters(batch_size            = 10,
                         learning_rate         = 0.01,
                         epochs                = 1000,
                         deterministic         = False,
                         training_attn_dropout = 0.05,

                         d_model               = 512 // 2,
                         seq_length            = 256 // 2,
                         vocabulary_size       = 1000 // 2,
                         extra_tokens          = ('<empty>',),
                         num_heads             = 8,
                         num_encoders          = 6,
                         num_decoders          = 6,

                         dataset               = 'wmt14_translate/de-en',
                         model_folder          = 'models',
                         vocabulary_prefix     = 'sentencepiece',
                         )

