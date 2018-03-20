import numpy as np
import scipy.linalg as la
import scipy
import random
import math
import networkx as nx
import copy

#The objective function
def phi(a, x, b):
    return (0.5 * np.dot((A_X(a, x) - b).transpose(), A_X(a, x) - b))

#The gradient of the objective function
def dphi(a, x, b, npoints, n):
    y = A_X(a, x) - b
    T = npoints + n
    gradf = np.zeros((T, T))
    for ind, A in enumerate(a):
        gradf = gradf + y[ind] * A
    return gradf

#The scaled gradient of the objective function
def ddphi(a, x, b, npoints, n):
    y = A_X(a, x) - b

    T = npoints + n
    gradf = np.zeros((T, T))
    for ind, A in enumerate(a):
        gradf = gradf + y[ind] * A
    return np.matmul(np.matmul(x, gradf), x)

#Performs the operation AX as enumerated in the assignment
def A_X(a, x):
    element_wise = [np.multiply(A, x) for A in a]
    return np.array([A.sum() for A in element_wise])

#Performs steepest descent with projects in the PSD space
def steepest_descent(op, dop, a, xin, b, niter, npoints, n):
    x = np.copy(xin)
    for i in range(0, niter):

        grad = dop(a, x, b, npoints, n)
        alpha = 1.

        #Here, we perform the backtracking line search
        #but we also want the smallest eigenvalue of x^(k+1)
        #to be at least half of the smallest eigenvalue of
        #x^k
        e_val_new, e_vec = np.linalg.eig(x - alpha*grad)
        e_val_old, t = np.linalg.eig(x)
        ratio = np.amin(e_val_new)/np.amin(e_val_old)

        #Perform backtracking
        while((phi(a,x - alpha*grad, b) > phi(a, x, b)) \
              or ratio < 0.5):
            alpha *= 0.8
            e_val_new, t = np.linalg.eig(x - alpha*grad)
            e_val_old, t = np.linalg.eig(x)
            ratio = np.amin(e_val_new)/np.amin(e_val_old)

        #Here, we perform the eigendecomposition and project
        #back into the positive semidefinite space
        x -= alpha * grad
        e_val_new[e_val_new < 0] = 0
        x = np.dot(np.dot(e_vec.transpose(), np.diag(e_val_new)), e_vec)
        x = np.real(x)
    return x

#Performs steepest descent with LDL factorization
def steepest_descent_ldl(op, dop, a, xin, b, niter, npoints, n):
    x = np.copy(xin)
    for i in range(0, niter):
        grad = dop(a, x, b, npoints, n)
        alpha = 1.

        #Perform backtracking
        while(phi(a,x - alpha*grad, b) > phi(a, x, b)):
            if(alpha < 1e-9):
                break
            alpha *= 0.8
        x -= alpha * grad

        #Compute the LDL factorization and zero out
        #the eigenvalues
        (P, L, U) = la.lu(x)
        D = np.diag(np.diag(U))
        U /= np.diag(U)[:, None]
        d = copy.copy(np.diag(D))
        d = np.array([-elem if elem < 0 else elem for elem in d])
        D = np.diag(d)
        x = P.dot(L.dot(D).dot(L.transpose()))

    return x

#Performs steepest descent with only the six largest eigenvalues
#and vectors
def steepest_descent_largest_evals(op, dop, a, xin, b, niter, npoints, n):
    x = np.copy(xin)
    for i in range(0, niter):
        grad = dop(a, x, b, npoints, n)
        alpha = 1.

        #Perform backtracking
        while(phi(a,x - alpha*grad, b) > phi(a, x, b)):
            if(alpha < 1e-9):
                break
            alpha *= 0.8
        x -= alpha * grad

        #Compute the six largest eigenvalues and eigenvectors
        temp = np.zeros((npoints + n, npoints + n))
        evals = scipy.linalg.eigh(x, eigvals_only = True, \
                                  eigvals = (npoints + n - 6, npoints + n - 1))
        t, evecs = scipy.linalg.eigh(x)
        evecs = evecs[:, -6:]

        #zero out the eigenvalues
        evals[evals < 0] = 0

        #Reconstruct the X matrix
        for i, e_val in enumerate(evals):
            temp += e_val * np.outer(evecs[:,i], evecs[:,i])

        x = temp

    return x

n = 2
npoints = 30

#Generate the problem with 30 sensors
(a, p, adjacency) = generate_inside_hull(n, npoints)

# Compute the Euclidian distances to the anchor points
adjSize = len(p) + len(a)
asize = len(a)
d = []

#Compute the distance variable
for i in range(adjSize):
    for j in range(adjSize):
        if(j > i and adjacency[i][j] > 0 and i < asize):
            d.append((adjacency[i][j], j - asize, i, True))
        elif(j > i and adjacency[i][j] > 0):
            d.append((adjacency[i][j], i - asize, j - asize, False))

T = n + npoints

#Construct the constraints we are going to use
eyeConstraint = []
anchorConstraints = []
pointConstraints = []

for i in range(n):
    temp = np.zeros((T,T))
    temp[i][i] = 1
    eyeConstraint.append((temp, 1))

temp = np.zeros((T,T))
for i in range(n):
    for j in range(n):
        temp[i][j] = 1
eyeConstraint.append((temp, n))

for (distance, i, j, truth) in d:
    if truth:
        temp = np.zeros(npoints)
        temp[i] = -1.
        anchorConstraints.append((np.outer(np.append(a[j], temp), \
                                           np.append(a[j], temp)), \
                                           distance ** 2))
    else:
        tempi = np.zeros(npoints)
        tempj = np.zeros(npoints)
        tempi[i] = 1.
        tempj[j] = 1.
        temp = tempi - tempj
        corner = np.zeros(n)
        temp = np.append(corner, temp)
        pointConstraints.append((np.outer(temp,temp), distance ** 2))

matConstraints = eyeConstraint + anchorConstraints + pointConstraints

#Put the constraints inside the A terms and put the RHS of the
#constraints as part of the solutions in the vector b
A = [mat[0] for mat in matConstraints]
b = [mat[1] for mat in matConstraints]

#Generate a random X matrix, and zero our the identity portion
#to store our solution
X = np.random.rand(T,T)
X = np.dot(X, X.transpose())

for i in range(n):
    for j in range(n):
        if(i == j):
            X[i][j] = 1
        else:
            X[i][j] = 0

import time

#Perform the three different methods
print "Steepest Descent"
start = time.time()
soln = steepest_descent(phi, dphi, A, X, b, 200, npoints, n)
Solution = soln[0:n, n:n + npoints].transpose()
end = time.time()
print "Time to compute in seconds: ", end - start
print "Average RMSE: ", math.sqrt(np.linalg.norm(p - Solution)**2)

print "LDL Factorization"
start = time.time()
soln = steepest_descent_ldl(phi, ddphi, A, X, b, 200, npoints, n)
Solution = soln[0:n, n:n + npoints].transpose()
end = time.time()
print("Time to compute: ", end - start)
print "Average RMSE: ", math.sqrt(np.linalg.norm(p - Solution)**2)

print "6 Largest Eigenvalues and vectors"
start = time.time()
soln = steepest_descent_largest_evals(phi, ddphi, A, X, b, 200, npoints, n)
Solution = soln[0:n, n:n + npoints].transpose()
end = time.time()
print("Time to compute: ", end - start)
print "Average RMSE: ", math.sqrt(np.linalg.norm(p - Solution)**2)

#Draw convergence graph for steepest descent with LDL
#factorization
iterList = [i for i in range(0, 100, 5)]
RMSEGraph = []
for it in iterList:
    soln = steepest_descent_ldl(phi, dphi, A, X, b, it, npoints, n)
    Solution = soln[0:n, n:n + npoints].transpose()
    RMSEGraph.append(np.linalg.norm(p - Solution))

plt.plot(iterList, RMSEGraph)
plt.xlabel("Number of Iterations")
plt.ylabel("RMSE")
plt.show()
