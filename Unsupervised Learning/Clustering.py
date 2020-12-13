# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 02:30:48 2020

@author: Admin
"""
import random
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as mcolors
colors = list(mcolors.CSS4_COLORS.keys())
random.shuffle(colors)
[colors.remove(each) for each in ['whitesmoke', 'white', 'snow', 'mistyrose', 'seashell', 'linen', 'floralwhite', 'ivory', 'beige', 'lightyellow', 'lightgoldenrodyellow', 'mintcream', 'azure', 'aliceblue', 'ghostwhite', 'lavenderblush']]

class GaussianMixture(NamedTuple):
    mu: np.ndarray
    var: np.ndarray
    p: np.ndarray

def sampleFromGaussian(K = 3, min_mu = 20, max_mu = 80, var = 15):
    points = [] 
    mu_list = []
    for each in range(0, K):
        mu = np.random.randint(low = min_mu, high = max_mu, size=2)
        mu_list.append(mu)    
        points.append(np.array([(each, *np.random.normal(mu,np.sqrt(var))) for _ in range(0,100)]))    
    gm = GaussianMixture(np.array(mu_list), np.array([var]*3), [1]*3)
    points = np.array(points).reshape(-1,K)
    return points, gm

def plot(points, gm):
    K = len(gm.mu)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim((0,100))
    ax.set_ylim((0,100))
    for i, point in enumerate(points):
        arc = patches.Arc(xy = point[1:], width = 1, height = 1, color = colors[int(point[0])]) 
        ax.add_patch(arc)
    for gmIdx in range(0, K):
        circle = patches.Circle(xy = gm.mu[gmIdx], radius = np.sqrt(gm.var[gmIdx])*2.56, fill = False, edgecolor = colors[gmIdx])
        ax.add_patch(circle)
    plt.show()

class KMeansAlgorithm:
    def __init__(self, data, K):
        '''
            Parameters:
                data: (n, d) numpy array
                K: number of clusters
        '''
        self.data = data
        self.cluster = np.array([-1]*len(data))
        self.K = K
        # Randomly select K points to be representatives
        self.Z = data[np.random.choice(len(data), size = K)]
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum(np.square(point2-point1)))
    
    def assignPointsToClusters(self):
        for i, point in enumerate(self.data):
            self.cluster[i] = np.argmin([self.euclidean_distance(point1 = point, point2 = z) for z in self.Z])
    
    def findBestRepresentatives(self):
        for k in range(0, self.K):
            cluster_points = self.data[self.cluster == k]
            self.Z[k] = np.sum(cluster_points, axis = 0) / len(cluster_points)
    
    def total_cost(self):
        return sum([self.euclidean_distance(point1 = point, point2 = self.Z[self.cluster[i]])for i, point in enumerate(self.data)])
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim((0,100))
        ax.set_ylim((0,100))
        for i, point in enumerate(self.data):
            arc = patches.Arc(xy = point, width = 1, height = 1, color = colors[self.cluster[i]]) 
            ax.add_patch(arc)
        plt.show()
        
    def run(self):
        prev_cost = None
        cost = None
        iteration = 0
        while (prev_cost is None or prev_cost - cost > 1e-4):
            iteration += 1
            print('Iteration = ', iteration)
            prev_cost = cost
            self.assignPointsToClusters()
            self.findBestRepresentatives()
            cost = self.total_cost()
            self.plot()

def main():
    global points
    K = 3
    points, gm = sampleFromGaussian(K = K)
    plot(points, gm)
    global kmeans
    data = points[:, 1:]
    kmeans = KMeansAlgorithm(data = data, K = K)
    kmeans.run()
    
if __name__ == "__main__":
    main()