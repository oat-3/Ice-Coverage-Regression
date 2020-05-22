# P7: COVID-19 Growth Trend Clustering
# ten_hundred.py
# Name: Oat (Smith) Sukcharoenyingyong
# Net ID: sukcharoenyi@wisc.edu
# CS login: sukcharoenyingyong

import numpy as np
import csv
import math

# todo: takes in a string with a path to a CSV file formatted as in the link above,
#  and returns the data (without the lat/long columns but retaining all other columns)
#  in a single structure.
def load_data(filepath):
    data = dict()
    dictlist = [ ]
    with open(filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = list(csv_reader)
        # header, keys for the dictionary
        header = csv_reader[0]
        # values in multiple dictionaries
        covidata = csv_reader[1:]

        # go through a list of values
        for row in covidata:
            # go through one individual value
            for i in range(len(row)):
                # ignore lat and long keys and values
                if i == 2 or i == 3:
                    continue
                # the values that are numbers should be integers
                if i > 3:
                    row[i] = int(row[i])
                # create dictionary with key header[i] and value row[i]
                data[header[i]] = row[i]
            # add the dictionary into a list
            dictlist.append(data.copy())
            data = dict()
    return dictlist


# todo: takes in one row from the data loaded from the previous function,
#  calculates the corresponding x, y values for that region as specified in the video,
#  and returns them in a single structure.
def calculate_x_y(time_series):
    values = list(time_series.values())
    # get rid of Province/State and Country/Region values
    values.pop(0)
    values.pop(0)
    # make the most recent date's data first, we want to loop through from
    # most recent date to least recent date
    values.reverse()
    # number of cases on the latest date in the csv file
    n = values[0]
    nten = n / 10
    nhundred = n / 100
    # initialize the index of day n/10 and day n/100 to NaN
    tenIdx = float('nan')
    hundredIdx = float('nan')



    for i in range(0, len(values) - 1):
        if values[i] > nhundred and values[i+1] <= nhundred:
            # find index of the day n/100
            hundredIdx = i + 1
            break
    for i in range(0, len(values) - 1):
        if values[i] > nten and values[i+1] <= nten:
            # find index of the day n/10
            tenIdx = i + 1
            break
    # calculate x and y using the index of days n, n/10, n/100
    x = tenIdx
    y = hundredIdx - tenIdx

    return (x,y)


# todo: performs single linkage hierarchical agglomerative clustering on the regions
#  with the (x,y) feature representation,  and returns a data structure
#  representing the clustering.
def hac(dataset):
    # list of clusters
    cluster = [ ]
    # list of numbers corresponding to the cluster
    num = [ ]
    # list of size of the clusters
    size = [ ]
    # take only the data points that do not contain NaN
    for i in range(len(dataset)):
        if math.isnan(dataset[i][0]) or math.isnan(dataset[i][1]):
            continue
        else:
            cluster.append([dataset[i]])

    m = len(cluster)
    for i in range(m):
        # number the original clusters
        num.append(i)
        # set size of all individual clusters to 1
        size.append(1)

    arr = [ ]
    # one iteration is equal to forming one new cluster from two clusters
    for i in range(m - 1):
        # initialize minimum distance to infinity
        min = float('inf')
        # compare each data point in each cluster with each other, j and k selects two clusters
        # l and n selects the data points in those two clusters j and k to perform euclidean
        # distance between two data points
        for j in range(len(cluster) - 1):
            for k in range(j + 1, len(cluster)):
                for l in range(len(cluster[j])):
                    for n in range(len(cluster[k])):
                        # euclidean distance of two points
                        eucD = math.sqrt(
                            (cluster[j][l][0] - cluster[k][n][0]) ** 2 + (cluster[j][l][1] - cluster[k][n][1]) ** 2)
                        if eucD < min:
                            # find the minimum distance
                            min = eucD
                            # the indexes of the two data points that are minimum distance from each other
                            idx0 = j
                            idx1 = k

        # add the two clusters together to form one new cluster
        tuple = cluster[idx0] + cluster[idx1]
        # add the new cluster into the cluster list
        cluster.append(tuple.copy())
        # the number corresponding to the new cluster
        num.append(m + i)
        # the size of the new cluster
        newSize = size[idx0] + size[idx1]
        # record the size of the new cluster
        size.append(newSize)
        # add the data into the array
        arr.append([num[idx0], num[idx1], min, newSize])

        # remove the previous two clusters that were used to form a new one
        cluster.pop(idx0)
        cluster.pop(idx1 - 1)
        num.pop(idx0)
        num.pop(idx1 - 1)
        size.pop(idx0)
        size.pop(idx1 - 1)

    # convert into numpy matrix
    nparr = np.array(arr)
    matrix = np.asmatrix(nparr)
    return matrix





