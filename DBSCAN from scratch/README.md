PROBLEM 5: DBSCAN on toy-neighborhood data
You are to cluster, and visualize, a small dataset using DBSCAN epsilon = 7.5, MinPts = 3. You have been provided a file, dbscan.csv, that has the following columns for each point in the dataset:

cluster originally empty, provided for your convenience pt a unique id for each data point
x point x-coordinate
y point y-coordinate
num_neighbors: number of neighbors, according to the coordinates above neighbors the id’s of all neighbors within
As you can see, a tedious O(n^2) portion of the work has been done for you. Your job is to execute, point-by-point, the DBSCAN algorithm, logging your work.

PROBLEM 6: DBSCAN on toy raw data
Three toy 2D datasets are provided (or they can be obtained easily with scikit learn) circles; blobs, and moons. Run your own implementation of DBSCAN on these, in two phases.

PROBLEM 7: DBSCAN on real data
Run the DBSCAN algorithm on the 20NG dataset, and on the FASHION dataset, and the HouseHold dataset (see papers), and evaluate results. You need to implement both phases (1) neighborhoods creation, (2) DBSCAN. Explain why/when it works, and speculate why/when not. You need to trial and error for parameters epsilon and MinPts

DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN DBSCAN Revisited:Mis-Claim, Un-Fixability, and Approximation

EXTRA CREDIT: Using class labels (cheating), try to remove/add points in curate the set for better DBSCAN runs
