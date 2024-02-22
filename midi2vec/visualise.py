import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(G):
    """
    Visualize the graph G using matplotlib.
    """
    plt.figure(figsize=(12, 12))  # Set the figure size as needed
    pos = nx.spring_layout(G)  # Positions for all nodes

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=70)

    # Edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Labels (optional, can be omitted for large graphs)
    nx.draw_networkx_labels(G, pos, font_size=5)

    plt.axis('off')  # Turn off the axis
    plt.show()
