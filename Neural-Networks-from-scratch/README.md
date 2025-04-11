## Implemented Neural Network for classification with mini batch gradient descent using numpy<br>
Dataset: MNIST Handwritten digit images (0-9) <br>
Training images: 50,000 <br>
Testing images: 10,000 <br>
Image size: 28x28 <br>

## Architecture
### 1 hidden layer
### Loss function: Cross-entropy loss
### Activation function:
- Sigmoid (Hidden layer)
- Softmax (Output layer)
## Hyperparameters
### Learning Rate: 5
### Number of hidden units: 300
### Batch size: 1000
### Epochs: 30
## Backpropagation formulae
### High Level steps
- Find derivative wrt activation layer (Z)
- Find derivative wrt pre-activation layer (A)
- Find derivative wrt Weights (W) and bias (B)
### IMPORTANT POINTS
![alt text](image-1.png)
#### This simple formula results from these facts:
- Softmax in the final layer
- Cross‐entropy as the loss function
- One‐hot labels 
#### Derivative of sigmoid
![alt text](image-2.png)
### Final formulae that I implemented
![alt text](image.png)