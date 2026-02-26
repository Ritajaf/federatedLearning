# Updated instantiation of the DeepSC model

# Assuming that we have already loaded the training configuration and vocabulary size.
# Sample configuration might look like this:
training_config = {
    'num_layers': 4,
    'd_model': 128,
    'num_heads': 8,
    'dff': 512
}
vocabulary_size = len(loaded_data)  # Assuming loaded_data is the dataset from which we extract the vocabulary

# Instantiate the DeepSC model with the configuration parameters
model = DeepSC(
    num_layers=training_config['num_layers'],
    d_model=training_config['d_model'],
    num_heads=training_config['num_heads'],
    dff=training_config['dff'],
    vocab_size=vocabulary_size
)