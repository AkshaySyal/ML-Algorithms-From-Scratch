import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from KMeans import KMeans
import pickle
from sklearn.datasets import fetch_20newsgroups

def transform_mnist(data):
       transformed_data = data.reshape(data.shape[0],-1)
       transformed_data[transformed_data>0] = 1
       return transformed_data

def transform_fashion(data):
       transformed_data = data.reshape(data.shape[0],-1)
       return transformed_data
   
def transform_news_groups(data):
        pass

def visualizeImg(img,lbl,reshaped=False):
        i = 3
        if(reshaped):
               plt.imshow(img.reshape((28,28)), cmap='viridis', interpolation='nearest')
        else:
               plt.imshow(img, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title(f'Image of {lbl}')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

if __name__ == '__main__':
       
       #imgs = idx2numpy.convert_from_file("Datasets/Fashion /t10k-images-idx3-ubyte")
       #imgs_copy = np.copy(imgs)
       #lbls = idx2numpy.convert_from_file("Datasets/Fashion /t10k-labels-idx1-ubyte")
       #lbls_copy = np.copy(lbls)
       filename = 'Datasets/20 NG/dataset.pkl'

       with open(filename, 'rb') as file:
             ng_data = pickle.load(file)
        
       newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
       lbls = newsgroups_test.target
       target_names = newsgroups_test.target_names

       text = np.copy(ng_data)
       labels = np.copy(lbls)
       
       kmeans = KMeans(k=40,dist_type='Euclidean',iters=10,num_of_true_lbls=20)
       kmeans.fit(data=text,true_lbls=labels)
       kmeans.evaluteClustering()
    
    #transformed_imgs = transform_mnist(imgs_copy)    
    #transformed_imgs = transform_fashion(imgs_copy)
#     i=121
#     visualizeImg(img=imgs_copy[i],lbl=lbls_copy[i])
#     visualizeImg(img=transformed_imgs[i],lbl=lbls[i],reshaped=True)
    
    #kmeans = KMeans(k=20,dist_type='Euclidean',iters=25,num_of_true_lbls=10)
    #kmeans.fit(data=transformed_imgs,true_lbls=lbls_copy)    
    #kmeans.evaluteClustering()

    # MNIST k=10
    #Objective function value: 1485053.0, Purity: 0.2048, Gini Average: 0.8636386695690841
    
    # MNIST k=5
    #Objective function value: 1500369.0, Purity: 0.189, Gini Average: 0.8555777734123303
    
    # MNIST k=20
    #Objective function value: 1471162.0, Purity: 0.1993, Gini Average: 0.8580059860829399
    
    # FASHION k=10
    #Objective function value: 489841403.0, Purity: 0.1618, Gini Average: 0.8811556562125022
    
    # FASHION k=5
    #Objective function value: 770137032.0, Purity: 0.2008, Gini Average: 0.8542479519654073
    
    # FASHION k=20
    #Objective function value: 477813985.0, Purity: 0.1627, Gini Average: 0.8795803409135767
    
    # 20NG k=20
    # Objective function value: 222293.24200732622, Purity: 0.08470525756771109, Gini Average: 0.9328196371857366
    
    # 20NG k=10
    # Objective function value: 248522.20982834254, Purity: 0.07514604354753053, Gini Average: 0.9416480514077292
    
    # 20NG k=40
    # RuntimeWarning: invalid value encountered in divide
    # self.centroids[k] = self.pi.T[k] @ self.data / sum(self.pi.T[k])
    # Objective function value: nan, Purity: 0.09240573552841211, Gini Average: 0.930551486020236
    # probably one of the cluster didn't get any points near it. Hence the corresponding col in membership mat is zero
    
#     centroids = kmeans.centroids
#     for i in range(len(centroids)):
#            visualizeImg(centroids[i],0,reshaped=True)
     
    

    
    
    

    

   