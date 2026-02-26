import os
import json
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import SNR_to_noise, greedy_decode, SeqtoText, Channels
from tqdm import tqdm
from sklearn.preprocessing import normalize
from w3lib.html import remove_tags
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/scratch/8388258-fmi15/europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='/scratch/8388258-fmi15/europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='/scratch/8388258-fmi15/europarl/checkpoints/deepsc-THz-snr-200-250/checkpoint_150', type=str)
parser.add_argument('--channel', default='THz', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
#parser.add_argument('--bert-config-path', default='/content/drive/MyDrive/bert_model/bert_config.json', type=str)
#parser.add_argument('--bert-checkpoint-path', default='/content/drive/MyDrive/bert_model/bert_model.ckpt', type=str)
#parser.add_argument('--bert-dict-path', default='/content/drive/MyDrive/bert_model/vocab.txt', type=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Similarity():
    def __init__(self, model_name='all-mpnet-base-v2'):
        # Load Sentence-BERT model
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, real, predicted):
        # Encode sentences into embeddings
        embeddings1 = self.model.encode(real, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        embeddings2 = self.model.encode(predicted, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Compute cosine similarity
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        # Extract diagonal for one-to-one sentence similarity
        similarity_scores = cosine_scores.diagonal().cpu().numpy()
        return similarity_scores.tolist()

def performance(args, SNR, net):
    similarity = Similarity()
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score2 = []
    printed_at_snr = False  # Track if the FIRST sentence pair is printed at SNR = 0 dB
    net.eval()
    channels = Channels()
    with torch.no_grad():
        for snr in tqdm(SNR):
            word = []
            target_word = []
            noise_std = SNR_to_noise(snr)

            for sents in test_iterator:
                sents = sents.to(device)
                target = sents

                out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                    start_idx, args.channel,channels)

                sentences = out.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, sentences))
                word = word + result_string

                target_sent = target.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, target_sent))
                target_word = target_word + result_string

                # Print the first transmitted and received sentence only at SNR = 0 dB
                if not printed_at_snr and snr == 0 and len(result_string) > 1:
                    transmitted = target_word[1]
                    received = word[1]
                    sim_score = similarity.compute_similarity([transmitted], [received])[0]

                    print(f"SNR: {snr} dB")  # Add SNR value
                    print(f"First Transmitted Sentence: {transmitted}")
                    print(f"First Received Sentence: {received}")
                    print(f"SBERT Similarity Score: {sim_score:.4f}")
                    printed_at_snr = True  # Ensure we print only once at SNR = 0

            sim_score = similarity.compute_similarity(target_word, word)
            score2.append(np.mean(sim_score))

    return score2



def plot_similarity_vs_snr(SNR, similarity_score, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(SNR, similarity_score, marker='o', linestyle='-', label='Sentence-BERT Similarity')
    plt.title('Sentence-BERT vs SNR', fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('SBERT Score', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,50,100,150,200,250,300]

    args.vocab_file = '/scratch/8388258-fmi15/europarl/vocab.json'
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    model_paths = [os.path.join(args.checkpoint_path, fn) for fn in os.listdir(args.checkpoint_path) if fn.endswith('.pth')]
    model_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    model_path = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    print(f"Model loaded successfully from {model_path}.")

    sim_score = performance(args, SNR, deepsc)
    print("Similarity Scores:", sim_score)

    plot_similarity_vs_snr(SNR, sim_score, '/scratch/8388258-fmi15/SBERT_vs_SNR_trained_in_THz/snr-200-250-150EPOCH.png')

