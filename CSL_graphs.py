import networkx as nx
import itertools
import matplotlib.pyplot as plt
import numpy as np




def generate_csl(n,k1,k2):
    """function to generate a csl graph with a skip length of k1 and k2.
Skip length denotes here the number of skipped nodes in the graph"""
    G = nx.Graph()
    H = nx.Graph()
    #generate arrays
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

    #create direct neighbor edges
    for i in range(0,n):
        if i==n-1:
            G.add_edge(i, 0)
            H.add_edge(i, 0)
        else:
            G.add_edge(i, i + 1)
            H.add_edge(i, i + 1)

    #creating skip edges edges for both graphs

    for i in range (0,n):
        G.add_edge(g_array[i],g_array[i+k1])
        H.add_edge(h_array[i],h_array[i+k2])
    return G,H




for n in range(15,16): #select number of nodes to evaluate CSL graphs on
    z = 1 #step size between k

    for k in range(3,6):#select k to evaluate on. skip links given by k, k+z
        G_csl, H_csl = generate_csl(n, k, k+z)
         #plot graphs
        subax1 =plt.subplot(121)
        pos1=nx.circular_layout(G_csl)
        nx.draw(G_csl,pos1,with_labels=True)
        subax2 = plt.subplot(122)
        pos2=nx.circular_layout(H_csl)
        nx.draw(H_csl,pos2,with_labels=True)
        plt.show()
        #calculate adj. and D^-1 matrix
        A_G_csl = nx.to_numpy_array(G_csl)


        A_H_csl = nx.to_numpy_array(H_csl)

        D_G_csl = np.diag([1/val for (node, val) in G_csl.degree()])
        D_H_csl = np.diag([1/val for (node, val) in H_csl.degree()])


        random_walk_cfi_G = []
        random_walk_cfi_H = []
        random_walk_csl_G = []
        random_walk_csl_H = []
        n_steps = 10
        for i in range(1, n_steps+1):
            #compute random walk matrix 

            random_walk_csl_G.append(np.linalg.matrix_power(np.matmul(D_G_csl,A_G_csl), i))
            random_walk_csl_H.append(np.linalg.matrix_power(np.matmul(D_H_csl,A_H_csl), i))
            print(random_walk_csl_G)
            print(random_walk_csl_H)

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
