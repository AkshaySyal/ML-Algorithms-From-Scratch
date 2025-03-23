import numpy as np

class KMeans:
    def __init__(self,k,dist_type,iters,num_of_true_lbls):
        self.k = k # number of clusters
        self.dist_type = dist_type
        self.iters = iters
        self.num_of_true_lbls = num_of_true_lbls
        self.pi = None
        self.data = None
        self.centroids = None
        self.true_lbls = None
    
    def distance(self,x,y):
        if(self.dist_type == 'Euclidean'):
            return np.linalg.norm(x-y)
        elif(self.dist_type == 'Cosine Similarity'):
            return np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))

    def fit(self,data,true_lbls):
        self.data = data
        self.true_lbls = true_lbls
        # initializing centroids
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[indices]
    
    def computePi(self):
        # for all data points find the closest centroid and update pi
        # reinitializing pi everytime it gets recomputed
        self.pi = np.zeros((len(self.data),self.k), dtype=int)

        for i in range(len(self.data)):
            dist = self.distance(self.data[i],self.centroids[0])
            closest_centroid_idx = 0
            for centroid_idx in range(1,len(self.centroids)):
                if(self.distance(self.data[i],self.centroids[centroid_idx]) < dist):
                    dist = self.distance(self.data[i],self.centroids[centroid_idx])
                    closest_centroid_idx = centroid_idx

            self.pi[i][closest_centroid_idx] = 1
    
    def computeCentroids(self):
        # for all k clusters
        # pi[i] (reshaped to 1xN) is multiplied with Xi (NxD)
        # normalized by num of data points in cluster k i.e. sum(pi[i])
        for k in range(self.k):
            self.centroids[k] = self.pi.T[k] @ self.data / sum(self.pi.T[k])
    
    def predict(self):
        # returns cluster lbl allocated to each data point 
        iters = 0
        self.computePi()
        self.computeCentroids()
        old_objective_value = float('inf')
        new_objective_value = self.kmeansObjective()

        while iters < self.iters and abs(old_objective_value - new_objective_value) > 1e-6:
            self.computePi()
            if iters != self.iters - 1:
                self.computeCentroids()
            
            old_objective_value = new_objective_value
            new_objective_value = self.kmeansObjective()
            iters += 1
        
        print(f'Objective function value: {new_objective_value}')
        return np.argmax(self.pi, axis=1)


    def kmeansObjective(self):
        distances_squared = np.sum((self.data[:, np.newaxis] - self.centroids) ** 2, axis=2) # NxK matrix: Dist of each pt with each centroid
        filtered_distances = distances_squared * self.pi # Element wise multiplication of distances_sq with membership matrix
        return np.sum(filtered_distances) # Sum of all filtered distances
    
    def evaluteClustering(self):
        # Need to make confusion matrix of algorithm determined cluster indices (row) vs true cluster indices (column)
        # purity = sum of row wise max / total data points
        # Gini index for a row (algorithm determined cluster) [Gj] = 1- sum of(mij/Mj)^2 [i from 1 to number of true cluster]
        # Gini average = Gj * Mj/ total data points

        # creating confusion matrix
        algo_det_lbls = self.predict() # array of shape 1xN
        cm = np.zeros((self.k,self.num_of_true_lbls), dtype=int)

        for i in range(len(algo_det_lbls)):
            algo_det_lbl = algo_det_lbls[i]
            true_lbl = self.true_lbls[i] 
            cm[algo_det_lbl][true_lbl] += 1

        Pj_sum = np.sum(np.max(cm,axis=1))
        print(f"Purity: {Pj_sum/len(algo_det_lbls)}")

        Gj = []
        Mj = np.sum(cm,axis=1) # number of data points per cluster
        for i in range(len(cm)):
            mij = np.sum(cm[i] ** 2)
            if(Mj[i] == 0):
                Gj.append(0)
            else:
                Gj.append(1-(mij/Mj[i]**2))
            
        gini_avg = np.sum(Gj*Mj)/len(algo_det_lbls)
        print(f"Gini Average: {gini_avg}")




        



