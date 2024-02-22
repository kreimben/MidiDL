import torch
from torch.utils.data import Dataset


class Node2vecDataset(Dataset):
    def __init__(self, G):
        embeddings = {word: G.wv[word] for word in G.wv.key_to_index}

        self.words = list(embeddings.keys())
        self.embeddings = [embeddings[word] for word in self.words]

        vocab_size = len(G.wv.index_to_key)
        embedding_dim = G.vector_size

        # Initialize an empty tensor for storing embeddings
        self.embeddings_matrix = torch.zeros((vocab_size, embedding_dim))

        # Fill the embeddings matrix with the vectors from the Word2Vec model
        for i, word in enumerate(G.wv.index_to_key):
            self.embeddings_matrix[i] = torch.from_numpy(G.wv[word])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        embedding = self.embeddings_matrix[idx]
        return word, embedding #torch.tensor(embedding, dtype=torch.float)
