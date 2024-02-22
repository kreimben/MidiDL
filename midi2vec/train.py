import os

import networkx as nx
from gensim.models import Word2Vec
from node2vec import Node2Vec

embedding_name = 'midi_embeddings.emb'
saved_model_name = 'midi_model.model'
max_cpu_cores = os.cpu_count()


def train_node(G: nx.Graph, demensions=64, walk_length=64, num_walks=256, workers=max_cpu_cores - 1, window=16,
               min_count=1, batch_words=64):
    node2vec = Node2Vec(G, demensions, walk_length, num_walks, workers, seed=42)
    model = node2vec.fit(window, min_count, batch_words)

    model.wv.save_word2vec_format(embedding_name)
    model.save(saved_model_name)


def load_trained_node():
    embedding = Word2Vec.wv.load_word2vec_format(saved_model_name)
    model = Word2Vec.load(saved_model_name)

    return embedding, model
