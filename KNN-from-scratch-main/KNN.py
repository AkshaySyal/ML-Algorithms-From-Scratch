import numpy as np

class KNN:
    def __init__(self,k=3):
        self.k = k
        self.data = None
        self.labels = None
     
    def transform(self,data):
        data = data.reshape(data.shape[0],-1) # Flattening images, (60k,28,28) -> (60k,784)
        data[data > 0] = 1 # Boolean Indexing
        return data
    
    def predict(self,test):
        test = self.transform(test)
        predictions = []
        for t in test:
            predictions.append(self.predict_single(t))
        
        return predictions

    def fit(self,train,labels):
        train = self.transform(train)
        self.data = train
        self.labels = labels
  
    def predict_single(self,t):
        heap = list()

        for i in range (len(self.data)):
            dist = np.linalg.norm(self.data[i]-t)
            heap.append([dist,self.labels[i]])
            heap.sort(key=lambda x:x[0])
            if len(heap) > self.k:
                heap.pop()
        lbls = []
        for dist,lbl in heap:
            lbls.append(lbl)
        
        return max(set(lbls), key=lbls.count)