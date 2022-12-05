# epsilon is a small value, threshold
# for x from i to infinity
# do
#    for each state s
#    do
#        V_k[s] = max_a Σ_s' p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])
#    end
#    if  |V_k[s]-V_k-1[s]| < epsilon for all s
#        for each state s,
#        do
#            π(s)=argmax_a ∑_s′ p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])
#            return π, V_k
#        end
# end


# hej

# Bellman Equation for Q-Value
# Q(s,a) = Σ_s' p(s'|s,a)*(r(a,s,s') + γ*max_a' Q(s',a'))

# Bellman Equation for Value
# V(s) = max_a Σ_s' p(s'|s,a)*(r(a,s,s') + γ*V(s'))

# Bellman Equation for Policy
# π(s) = argmax_a Σ_s' p(s'|s,a)*(r(a,s,s') + γ*V(s'))

import numpy as np
from numpy import argmax


def expected_value(currentmatrix, formermatrix, i, j, newi, newj, gamma):
    return 0.8 * (formermatrix[newi][newj] + currentmatrix[newi][newj] * gamma) \
           + 0.2 * (formermatrix[i][j] + currentmatrix[i][j] * gamma)


def get_neighbours(matrix, i, j):
    neighbours = []
    if i > 0:
        neighbours.append((i - 1, j))
    if i < matrix.shape[0] - 1:
        neighbours.append((i + 1, j))
    if j > 0:
        neighbours.append((i, j - 1))
    if j < matrix.shape[1] - 1:
        neighbours.append((i, j + 1))
    return neighbours


def bellman(currentmatrix, formermatrix, i, j, gamma):
    neighbours = get_neighbours(currentmatrix, i, j)
    values = []
    for neighbour in neighbours:
        values.append(expected_value(currentmatrix, formermatrix, i, j, neighbour[0], neighbour[1], gamma))
    return max(values)


def bellman_policy(currentmatrix, formermatrix, i, j, gamma):
    neighbours = get_neighbours(currentmatrix, i, j)
    values = []
    for neighbour in neighbours:
        values.append(expected_value(currentmatrix, formermatrix, i, j, neighbour[0], neighbour[1], gamma))
    return neighbours[argmax(values)]


def repeated_bellman(currentmatrix, formermatrix, gamma, epsilon, iterations):

    nextmatrix = np.zeros((3, 3))

    for i in range(currentmatrix.shape[0]):
        for j in range(currentmatrix.shape[1]):
            nextmatrix[i][j] = bellman(currentmatrix, formermatrix, i, j, pow(gamma, iterations))

    if np.all(np.abs(nextmatrix - currentmatrix) < epsilon): # if the difference is smaller than epsilon
        print("Converged after " + str(iterations) + " iterations")
        # for each state s, do
        for i in range(currentmatrix.shape[0]):
            for j in range(currentmatrix.shape[1]):
                #π(s)=argmax_a ∑_s′ p(s′|s,a)*(r(a,s,s′) + γ*V_k−1[s′])
                print("Policy for state " + str((i, j)) + " is: " + str(bellman_policy(nextmatrix, currentmatrix, i, j, pow(gamma, iterations))))

        return nextmatrix
    iterations += 1

    if iterations > 900:
        print("Did not converge after 900 iterations")
        return nextmatrix

    return repeated_bellman(nextmatrix, currentmatrix, gamma, epsilon, iterations)

empty = np.zeros((3, 3))

# create 3x3 matrix
matrix = np.zeros((3, 3))
# set goal
matrix[1][1] = 10

matrix = (repeated_bellman(matrix, empty, 0.9, 0.003, 0))

# display nice matrix with 2 decimals
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print(matrix)

















