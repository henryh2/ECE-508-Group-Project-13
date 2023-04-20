
# Local Clustering Coefficent (LCC)

Finds the local clustering coefficent for each vertex in a graph, which can be done by finding all triangles that include the node (T), finding the outbound degree of the node (D), and commputing the coefficent as:

LLC = 2*T/(D * (D - 1))

# Community Detection by Label Propogation

1. Label all nodes with unique identifier.

2. Update all nodes's labels with the majority label of its neighbors (outgoing & incoming)

    a. Histogram the labels of all the neighbors for this node.

    b. Find the label with the large frequency amongst the node's neighbors (tiebreak on lower label value).

    c. Set this node's label to the most frequent label.

3. Repeat step 2 for max iterations (can oscillate on bipartite graphs);
