import numpy as np
import random
from collections import Counter


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def preprocess(text):
    text = text.lower().split()
    return text


def build_vocab(tokens, min_count=1):
    counter = Counter(tokens)
    vocab = {w for w, c in counter.items() if c >= min_count}
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def generate_training_data(tokens, word2idx, window_size=2):
    data = []
    for i, word in enumerate(tokens):
        if word not in word2idx:
            continue
        center = word2idx[word]
        for j in range(max(0, i - window_size),
                       min(len(tokens), i + window_size + 1)):
            if j != i and tokens[j] in word2idx:
                context = word2idx[tokens[j]]
                data.append((center, context))
    return data


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=50, negative_samples=5, lr=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.lr = lr

        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    def train_pair(self, center_word, context_word):
        v_c = self.W_in[center_word]          
        u_o = self.W_out[context_word]        

        score_pos = np.dot(u_o, v_c)
        sig_pos = sigmoid(score_pos)
        loss = -np.log(sig_pos + 1e-10)

        grad_pos = sig_pos - 1  

        grad_v_c = grad_pos * u_o
        grad_u_o = grad_pos * v_c

        for _ in range(self.negative_samples):
            neg_word = random.randint(0, self.vocab_size - 1)
            if neg_word == context_word:
                continue

            u_k = self.W_out[neg_word]
            score_neg = np.dot(u_k, v_c)
            sig_neg = sigmoid(score_neg)

            loss -= np.log(1 - sig_neg + 1e-10)

            grad_neg = sig_neg

            grad_v_c += grad_neg * u_k
            self.W_out[neg_word] -= self.lr * grad_neg * v_c

        self.W_in[center_word] -= self.lr * grad_v_c
        self.W_out[context_word] -= self.lr * grad_u_o

        return loss

    def train(self, training_data, epochs=5):
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)
            for center, context in training_data:
                total_loss += self.train_pair(center, context)
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":

    text = """
    we are learning word embeddings
    word embeddings are useful for NLP
    we love machine learning and NLP
    """

    tokens = preprocess(text)
    word2idx, idx2word = build_vocab(tokens)

    training_data = generate_training_data(tokens, word2idx)

    model = Word2Vec(
        vocab_size=len(word2idx),
        embedding_dim=50,
        negative_samples=5,
        lr=0.025
    )

    model.train(training_data, epochs=10)

    word = "learning"
    if word in word2idx:
        vec = model.W_in[word2idx[word]]
        similarities = np.dot(model.W_in, vec)
        nearest = np.argsort(-similarities)[:5]
        print("\nNearest words to:", word)
        for idx in nearest:
            print(idx2word[idx])