from deep_sc import DeepSC

# Initialize model parameters
num_layers = 4
 d_model = 128
 num_heads = 8
dff = 512
dropout = 0.1
max_len = 30

# Load vocabulary from data_root directory
vocab_path = 'data_root/vocabulary.txt'
vocabulary = load_vocabulary(vocab_path)

# Instantiate the DeepSC model
model = DeepSC(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout, max_len=max_len, vocab=vocabulary)