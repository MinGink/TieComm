import networkx as nx
import matplotlib.pyplot as plt


def measure_strength(G, node_i, node_j):
    list1 = set(G.neighbors(node_i))
    list1.add(node_i)
    list2 = set(G.neighbors(node_j))
    list2.add(node_j)
    strength =  len(list1 & list2) / ((len(list1)  * len(list2)) ** 0.5)
    return strength




def graph_test(threshold):
    G = nx.binomial_graph(6, 0.5, seed=666 , directed=False)
    temp_g = nx.Graph()
    temp_g.add_nodes_from(G.nodes(data=False))

    for e in G.edges():
        strength = measure_strength(G, e[0], e[1])
        print(strength)
        G.edges[e[0], e[1]]['weight'] = strength
        if strength > threshold:
            temp_g.add_edge(e[0], e[1], weight=strength)

    S = [ list(c) for c in nx.connected_components(temp_g)]

    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=100, width=1)
    subax2 = plt.subplot(122)
    nx.draw(temp_g, pos=nx.spring_layout(temp_g), with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=100, edge_cmap=plt.cm.Blues, width=1)
    plt.show()
    print('test')


if __name__ == '__main__':
    threshold = 0.7
    graph_test(threshold)

