# P8: Ice Cover Regression
# ice_cover.py
# Name: Oat (Smith) Sukcharoenyingyong
# Net ID: sukcharoenyi@wisc.edu
# CS login: sukcharoenyingyong

import math
import random

# todo: takes no arguments and returns the data as described below in an n-by-2 array
def get_dataset():
    dataset = [ ]
    # first year is year 1855
    year = 1855
    with open("lake_Mendota_data.txt") as f:
        for days in f:
            # days that the lake is frozen for that year
            dataset.append([year, int(days)])
            year += 1
    return dataset


# todo: takes the dataset as produced by the previous function and
#  prints several statistics about the data; does not return anything
def print_stats(dataset):
    # number of data points
    num = (len(dataset))
    print(num)
    total = 0
    for i in range(num):
        total += dataset[i][1]
    # mean days lake frozen
    mean = total / num
    print("{:.2f}".format(mean))
    var = 0
    for i in range(num):
        var = var + (dataset[i][1] - mean)**2
    var = var / (num - 1)
    # standard deviation
    sd = math.sqrt(var)
    print("{:.2f}".format(sd))


# todo: calculates and returns the mean squared error on the dataset given fixed betas
def regression(beta_0, beta_1):
    data = get_dataset()
    total = 0
    # calculate MSE of dataset
    for i in range(len(data)):
        total = total + (beta_0 + (beta_1 * data[i][0]) - data[i][1])**2
    mse = total / len(data)
    return mse


# todo: performs a single step of gradient descent on the MSE and
#  returns the derivative values as a tuple
def gradient_descent(beta_0, beta_1):
    data = get_dataset()
    tot0 = 0
    tot1 = 0
    # perform gradient descent on MSE
    for i in range(len(data)):
        tot0 = tot0 + (beta_0 + (beta_1 * data[i][0]) - data[i][1])
        tot1 = tot1 + ((beta_0 + (beta_1 * data[i][0]) - data[i][1]) * data[i][0])
    g0 = (tot0 * 2) / len(data)
    g1 = (tot1 * 2) / len(data)
    return (g0, g1)


# todo: performs T iterations of gradient descent starting at
#  LaTeX: (\beta_0, \beta_1) = (0,0)( β 0 , β 1 ) = ( 0 , 0 )
#  with the given parameter and prints the results; does not return anything
def iterate_gradient(T, eta):
    beta_0 = 0
    beta_1 = 0
    # perform T iterations of gradient descent starting at beta_0 = 0
    # and beta_1 = 0
    for i in range(1, T+1):
        grad = gradient_descent(beta_0, beta_1)
        beta_0 = beta_0 - (eta * grad[0])
        beta_1 = beta_1 - (eta * grad[1])
        print(i, "{:.2f}".format(beta_0), "{:.2f}".format(beta_1), "{:.2f}".format(regression(beta_0, beta_1)))


# todo:  using the closed-form solution, calculates and returns the values of
#  LaTeX: \beta_0β 0 and LaTeX: \beta_1β 1 and the corresponding MSE as a three-element tuple
def compute_betas():
    data = get_dataset()
    totY = 0
    totX = 0
    # calculate total days and the total years
    for i in range(len(data)):
        totY += data[i][1]
        totX += data[i][0]
    # mean days and mean year
    meanY = totY / len(data)
    meanX = totX / len(data)
    top1 = 0
    bottom1 = 0
    # calculate beta_0 and beta_1 using closed-form solution
    for i in range(len(data)):
        top1 = top1 + ((data[i][0] - meanX) * (data[i][1] - meanY))
        bottom1 = bottom1 + ((data[i][0] - meanX)**2)
    beta_1 = top1 / bottom1
    beta_0 = meanY - (beta_1 * meanX)
    # calculate MSE
    mse = regression(beta_0, beta_1)
    return (beta_0, beta_1, mse)


# todo: using the closed-form solution betas, return the predicted number of ice days for that year
def predict(year):
    # get the beta values
    betas = compute_betas()
    # y = beta_0 + (beta_1 * x)
    prediction = betas[0] + (betas[1] * year)
    return prediction


# todo: normalizes the data before performing gradient descent, prints results as in function 5
def iterate_normalized(T, eta):
    data = get_dataset()
    totX = 0
    # normalize X
    for i in range(len(data)):
        totX += data[i][0]
    meanX = totX / len(data)
    var = 0
    for i in range(len(data)):
        var = var + ((data[i][0] - meanX)**2)
    var = var / (len(data) - 1)
    sd = math.sqrt(var)
    for i in range(len(data)):
        data[i][0] = (data[i][0] - meanX) / sd
    beta_0 = 0
    beta_1 = 0
    # perform T iterations of gradient descent starting at beta_0 = 0
    # and beta_1 = 0
    for j in range(1, T + 1):
        tot0 = 0
        tot1 = 0
        # perform normalized gradient descent on new data with normalized X
        for i in range(len(data)):
            tot0 = tot0 + (beta_0 + (beta_1 * data[i][0]) - data[i][1])
            tot1 = tot1 + ((beta_0 + (beta_1 * data[i][0]) - data[i][1]) * data[i][0])
        g0 = (tot0 * 2) / len(data)
        g1 = (tot1 * 2) / len(data)
        grad = (g0, g1)
        beta_0 = beta_0 - (eta * grad[0])
        beta_1 = beta_1 - (eta * grad[1])
        total = 0
        # calculate MSE using the new data with normalized X
        for i in range(len(data)):
            total = total + (beta_0 + (beta_1 * data[i][0]) - data[i][1]) ** 2
        mse = total / len(data)
        print(j, "{:.2f}".format(beta_0), "{:.2f}".format(beta_1), "{:.2f}".format(mse))


# todo: performs stochastic gradient descent, prints results as in function 5
def sgd(T, eta):
    data = get_dataset()
    totX = 0
    # normalize X
    for i in range(len(data)):
        totX += data[i][0]
    meanX = totX / len(data)
    var = 0
    for i in range(len(data)):
        var = var + ((data[i][0] - meanX) ** 2)
    var = var / (len(data) - 1)
    sd = math.sqrt(var)
    for i in range(len(data)):
        data[i][0] = (data[i][0] - meanX) / sd
    beta_0 = 0
    beta_1 = 0
    # perform T iterations of gradient descent starting at beta_0 = 0
    # and beta_1 = 0
    for j in range(1, T + 1):
        # perform stochastic gradient descent on new data with normalized X
        # random data point
        i = random.randint(0, len(data) - 1)
        tot0 = beta_0 + (beta_1 * data[i][0]) - data[i][1]
        tot1 = (beta_0 + (beta_1 * data[i][0]) - data[i][1]) * data[i][0]
        g0 = tot0 * 2
        g1 = tot1 * 2
        grad = (g0, g1)
        beta_0 = beta_0 - (eta * grad[0])
        beta_1 = beta_1 - (eta * grad[1])
        total = 0
        # calculate MSE using the new data with normalized X
        for i in range(len(data)):
            total = total + (beta_0 + (beta_1 * data[i][0]) - data[i][1]) ** 2
        mse = total / len(data)
        print(j, "{:.2f}".format(beta_0), "{:.2f}".format(beta_1), "{:.2f}".format(mse))

dataset = get_dataset()
print_stats(dataset)