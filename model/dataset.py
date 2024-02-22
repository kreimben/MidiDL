from torch.utils.data import Dataset


class Node2vecDataset(Dataset):
    def __init__(self, graph_embedding, labels):
        self.graph_embedding = graph_embedding
        self.labels = labels

    def __len__(self):
        return len(self.graph_embedding)

    def __getitem__(self, idx):
        return self.graph_embedding[idx], self.labels[idx]
