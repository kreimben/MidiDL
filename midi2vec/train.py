import networkx as nx
from node2vec import Node2Vec


def train_node(G: nx.Graph, demensions=64, walk_length=30, num_walks=200, workers=4, window=10, min_count=1,
               batch_words=4, save_result=True, embedding_name='midi_embeddings.emb',
               saved_model_name='midi_model.model'):
    node2vec = Node2Vec(G, demensions, walk_length, num_walks, workers)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    if save_result:
        model.wv.save_word2vec_format(embedding_name)
        model.save(saved_model_name)
