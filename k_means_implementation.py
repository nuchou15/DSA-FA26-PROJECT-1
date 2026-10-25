# import necessary libaries
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# create a custom dataset to use for k-means algorithm 
# comes with x and y coordinates and are also generated around 3 different centers to mimic real data
# X has the coords, y has the make_blobs group labels
X,y = make_blobs(n_samples = 500, n_features = 2, centers = 3, random_state = 23)
fig = plt.figure(0) # create a figure
plt.grid(True) # show grid lines
plt.scatter(X[:,0],X[:,1]) # create scatter plot
plt.show() # show figure

# initialize random centroids
k = 4 # choose random k value (number of clusters)
clusters = {} # empty dict to store info about each cluster
np.random.seed(23) # random seed so it's reproducible 
# create k initial centroids
for idx in range(k):
    # generate a random 2D coordinate for the intial centroid 
    center = 2*(2*np.random.random((X.shape[1],))-1) # generate coord between -2 and 2
    points = [] # create empty list to hold points
    # store centroid and empty list of points in a dict
    cluster = { 
        'center' : center, 
        'points' : [] 
    }

    # store this cluster in the main dictionary using idx as its key (which will be 0, 1, and 2)
    clusters[idx] = cluster
clusters 

plt.scatter(X[:,0],X[:,1])
plt.grid(True) 
# loop through every cluster to plot its initial centroid
for i in clusters:
    # retrieve center coords for cluster i
    center = clusters[i]['center']
    # plot the centroid as a red star 
    plt.scatter(center[0],center[1],marker= '*', c = 'red', s=150) # center 0 has the x coordinates, center 1 has the y coordinates
plt.show() # show plot

# define Euclidean distance function 
def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

# assign data points to their nearest cluster 
def assign_clusters(X, clusters):
    # visit each data point in the dataset
    for idx in range(X.shape[0]):
        dist = [] # empty list to store distances from data point to centroid
        curr_x = X[idx] # current data point's coordinates
        # calculate distance from this data point to every centroid
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center']) # measure the distance btwn the data point and center i
            dist.append(dis) # store distance
        curr_cluster = np.argmin(dist) # index of nearest centroid
        clusters[curr_cluster]['points'].append(curr_x) # assign current data point to nearest cluster
    return clusters

# update clusters based on proximity to dataset 
def update_clusters(X, clusters):
    # visit each cluster
    for i in range(k):
        points = np.array(clusters[i]['points']) # convert points into array
        # if statement to avoid calculating mean of empty array
        if points.shape[0] > 0:
            new_center = points.mean(axis =0) # calculate mean of the points
            clusters[i]['center'] = new_center # replace old centroid
            clusters[i]['points'] = [] # clear cluster points list 
    return clusters

# predicting the cluster for data points
def pred_cluster(X, clusters):
    pred = [] # create a list to store predicted cluster index for each point
    # visit every data point in dataset
    for i in range(X.shape[0]):
        dist = [] # create a list to store distances from this point to each centroid
        # calculate distance to nearest centroid
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# Repeat until convergence or maximum iterations
max_iters = 100

# Updating part (repeating process to improve centroid location)
for _ in range(max_iters):
    # save copies of current centroid location before updating
    old_centers = [clusters[i]['center'].copy() for i in range(k)]
    clusters = assign_clusters(X, clusters) # assign every data point to nearest centroid
    clusters = update_clusters(X, clusters) # move each centroid to the mean position of its assigned points
    # collect updated centroid locations
    new_centers = [clusters[i]['center'] for i in range(k)]
    # check if centroid locations stopped changing so we don't need to keep iterating
    if np.allclose(old_centers, new_centers):
        break
# assign each data point to nearest centroid 
pred = pred_cluster(X, clusters)

# show final results
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='viridis')
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red', s=150, label='Centroid' if i == 0 else None)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.grid(True)
plt.show()