import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from KNN import KNN
import time

def train():
    start_time = time.time()

    train_images = idx2numpy.convert_from_file("dataset/train-images.idx3-ubyte")
    train_images_copy = np.copy(train_images)
    train_labels = idx2numpy.convert_from_file("dataset/train-labels.idx1-ubyte")

    test_images = idx2numpy.convert_from_file("dataset/t10k-images.idx3-ubyte")
    test_images_copy = np.copy(test_images)
    test_labels = idx2numpy.convert_from_file("dataset/t10k-labels.idx1-ubyte")

    clf = KNN(k=3)

    clf.fit(train_images_copy,train_labels)

    predictions = clf.predict(test_images_copy[:100])
    count = 0
    for i in range(len(predictions)):
        if(predictions[i] == test_labels[i]):
            count += 1
    
    print(f"Accuracy: {count/len(predictions)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run the code: {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    train()