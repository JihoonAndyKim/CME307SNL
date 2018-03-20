# Import CVX and numpy libraries
import cvxpy as cvx
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import snap

#Plot the graph
def plotGraph(n, npoints, adjacency, a, p):
    if(n == 2):
        G=nx.Graph()

        for i in range(n+1):
            G.add_node(i,pos=a[i])

        for i in range(npoints):
            G.add_node(n+1+i,pos=p[i])

        pos=nx.get_node_attributes(G,'pos')

        nx.draw_networkx_nodes(G,pos,
                               nodelist=[i for i in range(n+1)],
                               node_color='r',
                               node_size=50,
                               alpha=0.8)
        nx.draw_networkx_nodes(G,pos,
                               nodelist=[i for i in range(n+1,npoints+n+1)],
                               node_color='b',
                               node_size=50,
                               alpha=0.8)
        for i in range(npoints+n+1):
            for j in range(n+1,npoints+n+1):
                if(adjacency[j,i] > 0.):
                    G.add_edge(i,j)

        nx.draw_networkx_edges(G,pos)
        plt.draw()  # pyplot draw()

res = []
#Random constraints on the anchor points that we are interested in
tests = [(3, 4, 3), (3, 4, 2), (3, 4, 1), (3, 4, 0), (2, 3, 2), \
         (2, 3, 1), (2, 3, 0), (1, 2, 1), (1, 2, 0), (0, 1, 0)]
#Perform 200 trials
for w in range(200):
    if(w % 50 == 0):
        print "Iteration: ", w

    #2 dimensions with the number of sensors ranging from
    #1 to 20 in our problem
    n = 2
    npoints = np.random.randint(1, 20)

    #Choose a random anchor constraint
    c = random.choice(tests)

    #Randomize whether we generate the points randomly or
    #all inside the hull. This is to capture varying clustering
    #coefficients
    HullOrNot = random.randint(0, 1)
    if HullOrNot:
        (a, p, adjacency) = generate_inside_hull(n, npoints, c[0], c[1], c[2])
    else:
        (a, p, adjacency) = generate(n, npoints, c[0], c[1], c[2])
    if q % 10 == 0:
        (a, p, adjacency) = generate(n, npoints, 3, 4, 3)
    if q % 5 == 0:
        (a, p, adjacency) = generate_inside_hull(n, npoints, 3, 4, 3)

    # Compute the Euclidian distances to the anchor points
    adjSize = len(p) + len(a)
    asize = len(a)
    d = []

    #Compute the d variable from the adjacency matrix
    for i in range(adjSize):
        for j in range(adjSize):
            if(j > i and adjacency[i][j] > 0 and i < asize):
                d.append((adjacency[i][j], j - asize, i, True))
            elif(j > i and adjacency[i][j] > 0):
                d.append((adjacency[i][j], i - asize, j - asize, False))

    T = n + npoints

    z = cvx.Semidef(T)

    #Construct the constraints for our problem
    eyeConstraint = []
    anchorConstraints = []
    pointConstraints = []

    for i in range(n):
        temp = np.zeros((T,T))
        temp[i][i] = 1
        eyeConstraint.append(temp)

    temp = np.zeros((T,T))
    for i in range(n):
        for j in range(n):
            temp[i][j] = 1
    eyeConstraint.append(temp)

    for (distance, i, j, truth) in d:
        if truth:
            temp = np.zeros(npoints)
            temp[i] = -1.
            anchorConstraints.append((np.outer(np.append(a[j], temp),\
                                               np.append(a[j], temp)),\
                                                distance))
        else:
            tempi = np.zeros(npoints)
            tempj = np.zeros(npoints)
            tempi[i] = 1.
            tempj[j] = 1.
            temp = tempi - tempj
            corner = np.zeros(n)
            temp = np.append(corner, temp)
            pointConstraints.append((np.outer(temp,temp), distance))

    matConstraints = anchorConstraints + pointConstraints

    #Another empty states list
    states = []

    #Perform SDP and solve the problem
    cost = cvx.norm(0)

    #The four constraints in the SDP relaxation problem
    #Note that the last constraint forces z to be SPD
    constr = []

    for i, mat in enumerate(eyeConstraint):
        if i < len(eyeConstraint) - 1:
            constr.append(cvx.sum_entries(cvx.mul_elemwise(mat, z)) == 1)
        else:
            constr.append(cvx.sum_entries(cvx.mul_elemwise(mat, z)) == n)

    for i, mat in enumerate(matConstraints):
        constr.append(cvx.sum_entries(cvx.mul_elemwise(mat[0], z)) ==  mat[1] ** 2)

    constr.append(z >> 0)

    #Add the constraints and cost function
    states.append(cvx.Problem(cvx.Minimize(cost), constr))

    #Solve the SDP relaxation problem
    prob = sum(states)
    prob.solve();

    #Using this solution, compute the RMSE value for this trial
    SDPSolution = z.value.A[0:n, n:n + npoints].transpose()
    RMSE = np.linalg.norm(SDPSolution - p) ** 2

    #Construct the graph in SNAP
    Graph = snap.TUNGraph.New()

    for i in range(npoints + len(a)):
        Graph.AddNode(i)

    for (distance, i, j, truth) in d:
        if truth:
            Graph.AddEdge(i, j +  npoints)
        else:
            Graph.AddEdge(i, j)

    #Compute the clustering coefficients, don't include
    #any graphs that have coefficients of 0 (poorly
    #generated graphs)
    GraphClustCoeff = snap.GetClustCf(Graph, -1)
    if GraphClustCoeff > 0.001:
        res.append([RMSE, GraphClustCoeff])

#Plot the result of our findings
x = np.asarray([elem for elem in zip(*res)[1]])
y = np.log(np.asarray([elem for elem in zip(*res)[0]]))

plt.scatter(x, y)
plt.xlabel("Clustering Coefficient")
plt.ylabel("log(RMSE)")
#Regression line
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '-', color = 'r')
plt.show()

n = 2
npoints = 7
c = random.choice(tests)
HullOrNot = random.randint(0, 1)
(a, p, adjacency) = generate(n, npoints, 3, 4, 3)

# Compute the Euclidian distances to the anchor points
Graph = snap.TUNGraph.New()

for i in range(npoints + len(a)):
    Graph.AddNode(i)

for (distance, i, j, truth) in d:
    if truth:
        Graph.AddEdge(i, j +  npoints)
    else:
        Graph.AddEdge(i, j)

GraphClustCoeff = snap.GetClustCf(Graph, -1)
print GraphClustCoeff
plotGraph(n, npoints, adjacency, a, p)

n = 2
npoints = 10
c = random.choice(tests)
HullOrNot = random.randint(0, 1)
(a, p, adjacency) = generate(n, npoints, 1, 2, 1)

# Compute the Euclidian distances to the anchor points
Graph = snap.TUNGraph.New()

for i in range(npoints + len(a)):
    Graph.AddNode(i)

for (distance, i, j, truth) in d:
    if truth:
        Graph.AddEdge(i, j +  npoints)
    else:
        Graph.AddEdge(i, j)

GraphClustCoeff = snap.GetClustCf(Graph, -1)
print GraphClustCoeff
plotGraph(n, npoints, adjacency, a, p)
