import numpy as np
from numpy.random import randint, randn, rand
from numpy.linalg import norm

import sys
import time

import pandas as pd

import matplotlib.pyplot as plt
import itertools
import random

def centers_old(x,k):
    print('Calculating costs')
    #select k random medoids
    print('Rows: ',x.shape[0],'Columns: ',x.shape[1])
    #Select k random items (medoids)
    #medoids=randint(x.shape[0],size=2)
    try:
        centers=random.sample(range(x.shape[0]), k)
    except ValueError:
        print('k exceeds the population size')
        return None
    print('centers: ',centers)
    return centers

def all_centers(x,k):
        # Initialize cost matrix for all the possible set of clusters
    dimensions=[x.shape[0]]*k
    total_costs = np.zeros(dimensions)
    #print('NOW ITERATING')    
    # Create all possible combinations of clusters and iterate them using PAM
    iterable = [np.arange(x.shape[0])]*k
    #print('iterable',iterable)
    for iter in itertools.product(*iterable):
        if len(set(iter)) != 1:
            #print('Sending: ', iter)
            # The centers look like this (position of the value in the X dataset)
            #(9, 7, 8, 1)
            #(9, 7, 8, 2)
            #(9, 7, 8, 3)
            #(9, 7, 8, 4)

            total_costs[iter], clusters=PAMalg(k,x,np.array(iter))
    #print(total_costs)#<===== PRINT COSTS
    return total_costs

def PAMalg(k, x, centers):   
    centers=np.array(centers)   
    if k!=centers.size:
        print('Number of clusters does not correspond to the number of given medoids')
        return None
    #select the k random medoids pointed by centers from x
    print('\n')
    print('-'*60)
    medoids=[x[i] for i in centers]
    print('medoids:',medoids)
    #calculate distances
    #create a matrix with distances
    distances=np.zeros((x.shape[0],k))

    for i ,h in zip(medoids,range(k)):#i contains the current medoids
        #print(h)
        for j in range(x.shape[0]):#j contains the indexes to each point in the array x
            #print('x[j]: ',x[j],' medoids: ',i)
            #Use Manhattan distances
            distance = sum(abs(x[j]-i))
            #print(distance)
            distances[j,h]=distance
    # Create matrix with all distances between data and selected centers
    #print('distances:\n',distances) #<===============PRINT DISTANCES
    # Initialize cluster vector
    c_vector= np.zeros((x.shape[0],1))
    # Returns the indices of the minimum values along an axis (every column is a different k cluster)
    c_vector = [np.argmin(distances[i]) for i in range(distances.shape[0])]
    #print('c_vector:\n',c_vector) #<================PRINT CLUSTERING VECTOR
    #Translate the cluster vector with the original data (find real elements from data set)
    clusters = find_objects_cluster(x,c_vector,k)
    cost=calculate_cost(distances, c_vector, k)
    total_cost=sum(cost.values())
    return total_cost, clusters


def calculate_cost(distances, c_vector,k):
    #Calculate the costs
    cost={}
    for i in range(k):
        #print([distances[j,c_vector[j]] for j in range(distances.shape[0]) if c_vector[j] == i]) #<===== PRINT CLUSTERIZED DISTANCES
        cost[i]=sum([distances[j,c_vector[j]] for j in range(distances.shape[0]) if c_vector[j] == i])
    print('cost per cluster',cost)
    print('Total cost:',sum(cost.values()))
    return cost
    

def find_objects_cluster(x,c_vector,k,printit=False):
    # This translates the clustering vector in order the find the elements corresponding to each cluster
    clusters={}
    for i in range(k):
        clusters[i]=[x[j] for j in range(x.shape[0]) if c_vector[j] == i]
    if printit == True:
        for cluster, val in clusters.items():
            print('Cluster: ', cluster)
            print(val)

        #print('Clusters found:\n',clusters)
    
    return clusters

def find_x_best_solution(costs_matrix,k,best=2):
    print('costs matrix',costs_matrix)
    unique_costs, indices, count=np.unique(costs_matrix, return_index=True, return_counts=True)
    #Find all costs, sort them
    print('unique',unique_costs)
    print('indices',indices)
    print('counts',count)
    print('size',unique_costs.size)
    if unique_costs.size <= best:
        return  None
    else:
        print('Looking for cost:',unique_costs[best])
        coords=np.where(costs_matrix==unique_costs[best])
        #print('Coordinates\n',coords)
        #print('Len Coordinates\n',len(coords))
        coords = np.array(coords)
        #Turn duples into a matrix and extract the first column, this is the first set of clusters
        #print('Coordinates', coords)
        first=coords[:,0]
        print('Cluster in:\n',first)
        return first
def find_medoids(x,c):
    medoids=[x[i] for i in c]
    #print('Medoids',medoids)
    return medoids

def obtain_clusters():
    # Use a dataset to extract gaussian clouds
    data=pd.read_csv('clouds.csv')
    print(data.shape)
    f1=data['V1'].values
    f2=data['V2'].values
    X=np.array(list(zip(f1,f2)))
    #X=X[1:3000:50,:]
    plt.plot(f1,f2,'.k')
    #plt.show()
    print(X)
    print(X.shape)
    return X

def plot_test(x,cluster, medoids,k):
    plt.figure()
    cols=['.r','.g','.b','.c','.m','.y','.k']
    colsc=['*r','*g','*b','*c','*m','*y','*k']
    for i in range(k):
        a=np.array(cluster[i])
        plt.plot(a[:,0],a[:,1],cols[i%7])
        plt.plot(medoids[i][0],medoids[i][1],colsc[6])
    #plt.show()

def experiments2d(x,k,expe):
    meds=centers_old(x,k)
    best_clusters = PAMalg(k,x,meds)
    second_best = best_clusters
    best_meds=meds
    second_meds=meds    
    print(meds)
    for i in range(expe-1):
        meds=centers_old(x,k)
        medoids=PAMalg(k,x,meds)
        if medoids[0] < best_clusters[0]:
            second_best = best_clusters
            best_clusters = medoids
            second_meds = best_meds
            best_meds = meds
        if medoids[0] < second_best[0] and medoids[0] > best_clusters[0]:
            second_best = medoids
            second_meds = meds

    print('\n','*'*60)
    print('Summary\n\n')
    print('Best medoids:\n', best_meds)
    print('Best medoids:\n', find_medoids(x,best_meds))
    print('best_cost\n', best_clusters[0])
    print('\n','*'*60,'\n')
    print('Second best medoids\n', second_meds)
    print('Second_best medoids:\n', find_medoids(x,second_meds))
    print('second_best_cost\n', second_best[0])

    plot_test(x,best_clusters[1], find_medoids(x,best_meds),k)
    plt.title('Best medoids\nCost: {0:.2f}  K: {1}'.format(best_clusters[0],k))
    plot_test(x,second_best[1], find_medoids(x,second_meds),k)
    plt.title('Second Best medoids\nCost: {0:.2f}  K: {1}'.format(second_best[0],k))
    plt.show()


if __name__ == '__main__':
    # Data for the first experiment
    x=np.array([[2,6],
                [3,4],
                [3,8],
                [4,7],
                [6,2],
                [6,4],
                [7,3],
                [7,4],
                [8,5],
                [7,6]]
                )
    start_time = time.time()
    k=3 #provisional number of medoids
    expe = 100
    experiments2d(x,k,expe)
    print("\n--- %s seconds ---" % (time.time() - start_time))

    #Import data for the second experiment
    start_time = time.time()
    x=obtain_clusters()
    k=3 #provisional number of medoids
    expe = 100
    experiments2d(x,k,expe)
    print("\n--- %s seconds ---" % (time.time() - start_time))
