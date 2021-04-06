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
    num_heads : int
    num_encoders : int
    num_decoders : int
