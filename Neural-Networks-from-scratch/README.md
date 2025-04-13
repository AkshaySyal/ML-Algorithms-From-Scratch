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
#### Derivative of softmax
![alt text](image-3.png)
#### Derivative of sigmoid
![alt text](image-2.png)
### Final formulae that I implemented
![alt text](image.png)
## Results
<img width="630" alt="Screenshot 2025-04-13 at 4 41 23 PM" src="https://github.com/user-attachments/assets/4e130096-27e5-43b4-883c-884062d92b9e" />
<img width="601" alt="Screenshot 2025-04-13 at 4 41 38 PM" src="https://github.com/user-attachments/assets/495fec0c-7a8a-4f73-9432-a29112372342" />
![Screenshot 2025-04-13 at 4 49 21 PM](https://github.com/user-attachments/assets/dccd5fd8-ade1-4031-9745-4359f8fa3a6e)

