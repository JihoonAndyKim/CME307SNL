# Import CVX and numpy libraries
import cvxpy as cvx
import numpy as np
import random
import copy
import math
import time
import networkx as nx
import matplotlib.pyplot as plt

#The objective function for steepest descent
def SNL(a, x, d):
    sum_obj = 0

    for (distance, i, j, truth) in d:
        if truth:
            sum_obj += ((np.linalg.norm(a[j] - x[i]) ** 2) - \
                        distance ** 2) ** 2
        else:
            sum_obj += (((np.linalg.norm(x[i] - x[j]) ** 2) - \
                         distance ** 2) ** 2)/2

    return sum_obj

# Here, we perform the gradient of the objective function
def dSNL(a, x, d, npoints, n):
    sum_x = np.zeros((npoints, n))

    for (distance, i, j, truth) in d:
        if truth:
            sum_x[i] += 4 * (np.linalg.norm(a[j] - x[i]) ** 2 - \
                             distance ** 2) * (-a[j] + x[i])
        else:
            sum_x[i] += 4 * (np.linalg.norm(x[i] - x[j]) ** 2 - \
                             distance ** 2) * (x[i] - x[j])

    return sum_x

# Updated steepest descent where we perform it twice for the two
# different sensors
def steepest_descent_2(op, dop, a, xin, d, niter, npoints, n):
    x = np.copy(xin)

    for i in range(0, niter):
            direction = dop(a, x, d, npoints, n)

            alpha = 1.

            for j in range(npoints):
                while(op(a, x  - alpha * direction, d) > op(a, x, d) \
                      - 0.5 * alpha * np.dot(direction[j], direction[j])):
                        if alpha < 1e-9:
                            break
                        alpha *= 0.9

            x -= alpha * direction

    return x

def SNL_noise(n, npoints, convex, niter, printOut = False, graph = False):
    #Add timing and running totals
    start = time.time()
    totalRMSE = 0

    for iteration in range(niter):
        #Generate different types of problems
        if convex:
            (a, p, adjacency) = generate(n, npoints)
        else:
            (a, p, adjacency) = generate_inside_hull(n, npoints)
        # Compute the Euclidian distances to the anchor points
        adjSize = len(p) + len(a)
        asize = len(a)
        d = []
        #Generate the d variable for storing edges and
        #information from the adjacency matrix
        for i in range(adjSize):
            for j in range(adjSize):
                if(j > i and adjacency[i][j] > 0 and i < asize):
                    d.append((adjacency[i][j], j - asize, i, True))
                elif(j > i and adjacency[i][j] > 0):
                    d.append((adjacency[i][j], i - asize, j - asize, False))

        # Construct the CVX variables to minimize
        x = [cvx.Variable(1) for i in range(len(d) * 2)]

        #The size of the our adjacency matrix
        T = n + npoints

        z = cvx.Semidef(T)

        #The following constructs the SNL problem with noise
        #constraints
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
                anchorConstraints.append((np.outer(np.append(a[j], temp), \
                                         np.append(a[j], temp)), distance))
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

        #The following is the new objective function
        #and the constraints for our SNL with noise problem
        cost = cvx.norm(sum(x))

        constr = []

        #Force the constraints as enumerated in the handout
        for i, mat in enumerate(eyeConstraint):
            if i < len(eyeConstraint) - 1:
                constr.append(cvx.sum_entries(cvx.mul_elemwise(mat, z)) == 1)
            else:
                constr.append(cvx.sum_entries(cvx.mul_elemwise(mat, z)) == n)

        for i, mat in enumerate(matConstraints):
            constr.append(cvx.sum_entries(cvx.mul_elemwise(mat[0], z)) + \
                          x[2*i] - x[2*i + 1] ==  mat[1] ** 2)
            constr.append(x[2*i] >> 0)
            constr.append(x[2*i + 1] >> 0)

        constr.append(z >> 0)

        #Add the constraints and cost function
        states.append(cvx.Problem(cvx.Minimize(cost), constr))

        #Solve the SDP relaxation problem
        prob = sum(states)
        prob.solve();

        #This is our solution to the problem which we will
        #use as the initial condition for our steepest
        #descent method
        SDPSolution = z.value.A[0:n, n:n + npoints].transpose()

        d2 = copy.copy(d)

        for k in range(len(d)):
            (distance, i, j, truth) = d[k]

            if not truth:
                d2.append((distance, j, i, truth))

        #Plot the convergence of the steepest descent method
        graphMSE = []
        iterList = [i for i in range(0, 200, 5)]
        if graph:
            for it in iterList:
                soln = steepest_descent_2(SNL, dSNL, a, SDPSolution, \
                                          d2, it, npoints, n)
                MSE = 0
                for i in range(npoints):
                    if printOut:
                        print("Sensor " + str(i) + " is located at " + \
                              str(soln[i]) + " and the actual value is " + str(p[i]))
                    MSE += np.linalg.norm(np.asarray(soln[i]) - np.asarray(p[i]))
                graphMSE.append(MSE)
            plt.plot(iterList, graphMSE)
            plt.xlabel("Number of Iterations")
            plt.ylabel("RMSE")
            plt.show()

        #Solve with steepest descent
        soln = steepest_descent_2(SNL, dSNL, a, SDPSolution, d2, 200, npoints, n)
        RMSE = 0
        for i in range(npoints):
            if printOut:
                print("Sensor " + str(i) + " is located at " + str(soln[i]) + \
                      " and the actual value is " + str(p[i]))
            RMSE += np.linalg.norm(np.asarray(soln[i]) - np.asarray(p[i])) ** 2
        totalRMSE += RMSE

    #Compute the total time and RMSE
    end = time.time()
    print "Total Time Elapsed (sec): ", end - start
    print "Average RMSE: ", math.sqrt(totalRMSE / niter)

#Plot and solve a 2D problem with outside the cnovex hull
SNL_noise(2, 10, True, 1, False, True)
