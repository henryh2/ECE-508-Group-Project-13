
# Local Clustering Coefficent (LCC)

Finds the local clustering coefficent for each vertex in a graph, which can be done by finding all triangles that include the node (T), finding the outbound degree of the node (D), and commputing the coefficent as:

LLC = 2*T/(D * (D - 1))
