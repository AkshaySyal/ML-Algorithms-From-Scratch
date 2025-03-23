# KNN-from-scratch
Implemented KNN from scratch using numpy on MNIST dataset (60k training, 10k test)

Given number of training samples is n each having d number of features

## Time & Space Complexity: 

Training: KNN has no explicit training. Hence O(1)

Prediction: Θ(n*d) 

Given a test sample its distance is calculated against n training samplex. Computing Euclidean distance takes Θ(d).
Store only k shortest distances in an array. After appending distance sort the array. Pop out last element if length of array exceeds k.

Space Complexity: Θ(n*d) 
Have to store the entire training data at run time during prediction

Test Accuracy: 0.7177 for k = 3
Time taken to run the code: 1413.8525 seconds

Accuracy: 0.7399 for k = 5
