import numpy

import numpy as np
from numpy import argmax

def bellman_function(currentmatrix, rewardmatrix, i, j, gamma):
    neighbours, actions = get_neighbours(currentmatrix, i, j)
    values = []
    for neighbour in neighbours:
        values.append(expected_value(currentmatrix, rewardmatrix, i, j, neighbour[0], neighbour[1], gamma))
    # get the index of the max value
    index = argmax(values)
    # return the max value and the action
    return values[index], actions[index]

def expected_value(currentmatrix, rewardmatrix, i, j, newi, newj, gamma):
    return 0.8 * (rewardmatrix[newi][newj] + currentmatrix[newi][newj] * gamma) \
           + 0.2 * (rewardmatrix[i][j] + currentmatrix[i][j] * gamma)

def get_neighbours(matrix, i, j):
    action = []
    neighbours = []
    if i > 0:
        neighbours.append((i - 1, j))
        action.append("W")
    if i < matrix.shape[0] - 1:
        neighbours.append((i + 1, j))
        action.append("E")
    if j > 0:
        neighbours.append((i, j - 1))
        action.append("S")
    if j < matrix.shape[1] - 1:
        neighbours.append((i, j + 1))
        action.append("N")
    return neighbours, action

def repeated_bellman(currentmatrix, rewardmatrix, gamma, epsilon, iterations):
    nextmatrix = np.zeros((3, 3))

    # policy matrix of the same size as the value matrix with the actions in a list
    policy_matrix = np.array([["", "", ""], ["", "", ""], ["", "", ""]], dtype=object)

    for k in range(iterations):
        currentmatrix = nextmatrix.copy()
        for i in range(currentmatrix.shape[0]):
            for j in range(currentmatrix.shape[1]):
                nextmatrix[i][j], action = bellman_function(currentmatrix, rewardmatrix, i, j, gamma)
                # add the action to the policy matrix, without replacing the previous actions
                policy_matrix[i][j] += action
        # check if the difference between the current matrix and the next matrix is less than epsilon for all the elements
        if np.all(np.abs(currentmatrix - nextmatrix) < epsilon):
            print("Converged at iteration ", k)
            return currentmatrix, policy_matrix

    print("Didn't converge after 1000 iterations")
    print(k)
    return currentmatrix, policy_matrix


def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(matrix[i][j], end=" ")
        print()

def print_policy(policy_matrix):
    for i in range(policy_matrix.shape[0]):
        for j in range(policy_matrix.shape[1]):
            print(policy_matrix[i][j], end=" ")
        print()

def main():
    reward_matrix = np.zeros((3, 3))
    reward_matrix[1][1] = 10
    currentmatrix = np.zeros((3, 3))
    gamma = 0.9
    epsilon = 0.00000001
    iterations = 100000
    value_matrix, policy_matrix = repeated_bellman(currentmatrix, reward_matrix, gamma, epsilon, iterations)
    print_matrix(value_matrix)
    print_policy(policy_matrix)
    str_array = np.array([["S", "S", "S"], ["S", "S", "S"], ["S", "S", "S"]], dtype=object)
    str_array[0][0] += ( "G")
    print(str_array)


main()


