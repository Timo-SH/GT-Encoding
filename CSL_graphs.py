import networkx as nx
import itertools
import matplotlib.pyplot as plt
import numpy as np

#cfi generation from https://github.com/LingxiaoShawn/KCSetGNN/blob/main/core/data_utils/cfi.py

def create_gadget(k):
    G = nx.Graph()
    # create as and bs
    for i in range(k):
        G.add_node(("a", i))
        G.add_node(("b", i))

    # create middle nodes
    elements = list(range(k))
    i = 0
    for n in range(0, k + 1, 2):
        for subset in itertools.combinations(elements, n):
            G.add_node(('m', i))
            for k in subset:
                G.add_edge(('m', i), ('a', k))
            for k in set(elements) - set(subset):
                G.add_edge(('m', i), ('b', k))
            i += 1
    return G


def create_cfi(k):
    gadget = create_gadget(k - 1)

    # create all gadgets
    G = nx.Graph()
    for i in range(k):
        tmp = gadget.copy()
        label_mapping = {node: (i, *node) for node in tmp.nodes}
        G.update(nx.relabel_nodes(tmp, label_mapping))

    # add label
    colors = {}
    i = 0
    for node in G.nodes():
        # all middle nodes have same color, a and b have same color
        c = (node[0], -1) if node[1] == 'm' else (node[0], node[-1])
        if c not in colors:
            colors[c] = i
            i += 1
        G.nodes[node]['x'] = colors[c]

    # create connections among all nodes
    index = [0] * k
    base_G = nx.complete_graph(k)
    for edge in base_G.edges():
        left, right = edge
        edge_a = (left, 'a', index[left]), (right, 'a', index[right])
        edge_b = (left, 'b', index[left]), (right, 'b', index[right])
        G.add_edge(*edge_a)
        G.add_edge(*edge_b)
        index[left] += 1
        index[right] += 1

    H = G.copy()
    H.remove_edge(*edge_a)
    H.remove_edge(*edge_b)
    H.add_edge(edge_a[0], edge_b[1])
    H.add_edge(edge_b[0], edge_a[1])
    # create correct node labels
    return nx.convert_node_labels_to_integers(G, ordering="sorted"), nx.convert_node_labels_to_integers(H, ordering="sorted")


def create_grohe_cfi(k):
    return nx.convert_node_labels_to_integers(grohe_cfi(k, False) ), nx.convert_node_labels_to_integers(grohe_cfi(k, True))


def grohe_cfi(k, hat=False):
    G = nx.Graph()
    base_G = nx.complete_graph(k)

    edges_hash = {}
    # create edge-based vertex and connections
    for i, edge in enumerate(base_G.edges()):
        edges_hash[edge] = i
        edges_hash[(edge[1], edge[0])] = i
        G.add_node(('e', 0, i))
        G.add_node(('e', 1, i))
        G.add_edge(('e', 0, i), ('e', 1, i))

    # create node-based vertex, and connection to edge-based vertex
    for i, node in enumerate(base_G.nodes()):
        adjacent_edges = base_G.edges(node)
        encoded_edges = [edges_hash[edge] for edge in adjacent_edges]
        start = 1 if i == 0 and hat else 0
        for n in range(start, len(encoded_edges) + 1, 2):
            for subset in itertools.combinations(encoded_edges, n):
                vertex = ('v', subset, node)
                G.add_node(vertex)
                for edge in encoded_edges:
                    if edge in subset:
                        G.add_edge(vertex, ('e', 1, edge))
                    else:
                        G.add_edge(vertex, ('e', 0, edge))

    # create color for all vertex
    colors = {}
    i = 0
    for node in G.nodes():
        # all middle nodes have same color, a and b have same color
        c = (node[0], node[-1])
        if c not in colors:
            colors[c] = i
            i += 1
        G.nodes[node]['x'] = colors[c]

    return G


def generate_csl(n,k1,k2):
    G = nx.Graph()
    H = nx.Graph()
    g_array = []
    add_array = []
    h_array = []
    for i in range (0,n):
        g_array.append(i)
        h_array.append(i)
        add_array.append(i)

    g_array = g_array + add_array
    h_array = h_array + add_array
    #creating nodes
    for i in range (0,n):
        G.add_node(i)
        H.add_node(i)


    for i in range(0,n):
        if i==n-1:
            G.add_edge(i, 0)
            H.add_edge(i, 0)
        else:
            G.add_edge(i, i + 1)
            H.add_edge(i, i + 1)

    #creating edges for both graphs
    """for i in range (0,n-k1):

        j = i + k1
        G.add_edge(i,j)
        #if i==n-k1-1:
        #    G.add_edge(i,0)
    print(G.number_of_edges())
    for i in range (0,n-k2):
        j = i + k2

        H.add_edge(i,j)
        if i==n-k2-1:
            G.add_edge(i,0)
    print(H.number_of_edges())"""

    for i in range (0,n):
        G.add_edge(g_array[i],g_array[i+k1])
        H.add_edge(h_array[i],h_array[i+k2])
    return G,H


#G, H = generate_csl(15,2,5)

"""subax1 =plt.subplot(121)
pos1=nx.circular_layout(G)
nx.draw(G,pos1,with_labels=True)
subax2 = plt.subplot(122)
pos2=nx.circular_layout(H)
nx.draw(H,pos2,with_labels=True)
plt.show()"""

def generate_custom_graph():
    G = nx.Graph()
    H = nx.Graph()
    for i in range(0,6):
        G.add_node(i)
        H.add_node(i)

    G.add_edge(0,4)
    G.add_edge(0, 5)
    G.add_edge(4, 2)
    G.add_edge(4, 5)
    G.add_edge(2, 5)
    G.add_edge(1, 5)
    G.add_edge(1, 4)
    G.add_edge(3, 4)
    G.add_edge(3, 5)

    H.add_edge(0, 1)
    H.add_edge(0, 4)
    H.add_edge(0, 5)
    H.add_edge(1, 4)
    H.add_edge(1, 5)
    H.add_edge(5, 2)
    H.add_edge(5, 3)
    H.add_edge(2, 3)
    H.add_edge(4, 3)
    H.add_edge(4, 2)

    return G, H


def compute_rw(graph_db, num_steps, rw_matrices):
    feature_vec = []
    for i in range(0,1 ):
        feature_vec.append(np.zeros(0, dtype=np.float64))

    offset = 0
    graph_ind = []

    for i in range(0,1 ):

        #print(graph_db.number_of_nodes())
        graph_ind.append((offset, offset + graph_db.number_of_nodes() -1))
        offset += graph_db.number_of_nodes()

    color_map = {}
    for i in range(0,1 ):
        for v in graph_db.nodes:
            color_map[v] = []

    c=1
    while c <= num_steps:
        colors = []
        for i in range(0,1):
            #get info from random walk depending on chosen steps
            j =0
            for v in graph_db.nodes:
                random_walk_vector = []
                for k in range(0, c):
                    random_walk_vector.append(rw_matrices[k][j,j])
                    #print(random_walk_vector)
                colors.append(hash(tuple(random_walk_vector)))
                #print(colors)
                #print(color_map[v])
                color_map[v].append(hash(tuple(random_walk_vector)))
                j = j+1
        #_, colors = np.unique(colors, return_inverse=True)
        q = 0
        for i in range(0,1):
            for v in graph_db.nodes:
                #color_map[v] = colors[q]
                q += 1

        max_all = int(np.amax(colors) + 1)

        feature_vec = color_map

        # Count how often each color occurs in each graph and create color count vector
        #feature_vec = [np.bincount(colors[index[0]:index[1] + 1], minlength=max_all) for i, index in
                           #enumerate(graph_ind)]

        c += 1

    #feature_vec = np.array(feature_vec)
    return feature_vec

# calc random walk matrices from graphs


for n in range(11,12):
    z = 1
    #G_cfi, H_cfi = create_cfi(n)
    #G_cfi, H_cfi = create_grohe_cfi(n)
    for k in range(3,4):
        G_csl, H_csl = generate_csl(n, k, k+z)
        #G_csl, H_csl = generate_custom_graph()

        #subax1 =plt.subplot(121)
        #pos1=nx.circular_layout(G_cfi)
        #nx.draw(G_cfi,pos1,with_labels=True)
        #subax2 = plt.subplot(122)
        #pos2=nx.circular_layout(H_cfi)
        #nx.draw(H_cfi,pos2,with_labels=True)
        #plt.show()

        subax1 =plt.subplot(121)
        pos1=nx.circular_layout(G_csl)
        nx.draw(G_csl,pos1,with_labels=True)
        subax2 = plt.subplot(122)
        pos2=nx.circular_layout(H_csl)
        nx.draw(H_csl,pos2,with_labels=True)
        plt.show()

        #A_G_cfi = nx.to_numpy_array(G_cfi)
        #A_H_cfi = nx.to_numpy_array(H_cfi)
        A_G_csl = nx.to_numpy_array(G_csl)
        print(A_G_csl)

        A_H_csl = nx.to_numpy_array(H_csl)
        print(A_H_csl)
    #D_G_cfi = np.diag([1/val for (node, val) in G_cfi.degree()])
    #D_H_cfi = np.diag([1/val for (node, val) in H_cfi.degree()])
        D_G_csl = np.diag([1/val for (node, val) in G_csl.degree()])
        D_H_csl = np.diag([1/val for (node, val) in H_csl.degree()])
    #print(A_G_cfi)
    #print(A_H_cfi)
    # always set graph db to 2 graphs from either csl or cfi

        random_walk_cfi_G = []
        random_walk_cfi_H = []
        random_walk_csl_G = []
        random_walk_csl_H = []
        n_steps = 10
        for i in range(1, n_steps+1):
            #random_walk_cfi_G.append(np.linalg.matrix_power(np.matmul(D_G_cfi, A_G_cfi), i))
            #random_walk_cfi_H.append(np.linalg.matrix_power(np.matmul(D_H_cfi, A_H_cfi), i))
            #random_walk_csl_G.append(np.linalg.matrix_power(np.matmul(D_G_csl, A_G_csl), i))
            #random_walk_csl_H.append(np.linalg.matrix_power(np.matmul(D_H_csl, A_H_csl), i))
            random_walk_csl_G.append(np.linalg.matrix_power(A_G_csl, i))
            random_walk_csl_H.append(np.linalg.matrix_power(A_H_csl, i))
            print(random_walk_csl_G)
            print(random_walk_csl_H)
        #feature_vec_cfi_G = compute_rw(G_cfi,n_steps, random_walk_cfi_G)
        #feature_vec_csl_G = compute_rw(G_csl, n_steps, random_walk_csl_G)
        #feature_vec_cfi_H = compute_rw(H_cfi,n_steps, random_walk_cfi_H)
        #feature_vec_csl_H = compute_rw(H_csl, n_steps, random_walk_csl_H)
        #print(feature_vec_cfi_H)
        #print(feature_vec_cfi_G)
        #print(feature_vec_csl_G)
        #print(feature_vec_csl_H)
        #c = 0
        #for i in range(0, len(feature_vec_csl_G)):
        #    for j in range(0, len(feature_vec_csl_G[i])):
        #        if feature_vec_csl_G[i][j] != feature_vec_csl_H[i][j]:
        #            c = 1
        #            print('discrepency found in step ' + str(j+1) + ' of the random walk at node ' + str(i))
        #if c==1:
            #print("discrepency found for n=" + str(n))
        #if c == 0:
        #    print("NO FOUND!!!! for n= " + str(n) + "and k=" + str(k))
        #    if k == (k+1) % n or k == -(k+z) % n:
        #        print("no found due to isomorphism at k=" + str(k) + " k+1=" + str(k+z) + " and n=" + str(n))
        #c = 0
        """"for i in range(0, len(feature_vec_cfi_G)):
            #print(feature_vec_cfi_G)
            for j in range(0, len(feature_vec_cfi_G[i])):
                if feature_vec_cfi_G[i][j] != feature_vec_cfi_H[i][j]:
                    c = 1
                    print('discrepency found in step ' + str(j+1) + ' of the random walk at node ' + str(i))
        if c==1:
            print("discrepency found for n=" + str(n))
        if c==0:
            print("NO FOUND!!!!")"""""