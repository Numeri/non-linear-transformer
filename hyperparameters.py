from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # Training parameters
    batch_size : int
    learning_rate : float
    epochs : int

    # Model parameters
    d_model : int
    seq_length : int
    num_heads : int
