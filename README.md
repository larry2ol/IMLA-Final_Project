# IMLA-Final-Project
Project: CNN Image Classification with Bayesian Optimization. 

Overview: The project utilizes Convolutional Neural Network (CNN) architecture for image classification tasks, with Bayesian Optimization being used to fine-tune hyperparameters like convolutional layers, kernel size, kernel number, and dropout rates.

Why This Project is Useful?
The project offers optimized performance through Bayesian Optimization, allowing for fine-tuned hyperparameters, flexibility for image classification tasks, and scalability for deployment in cloud and edge environments, making it suitable for various tasks.

What You Can Do With This Project
This project allows users to train a CNN model on their own image datasets, use it for various image classification tasks like medical imaging and object detection, and extend it for multi-class classification with minor modifications.

Getting Started
Getting Started requires Python 3.x, PyTorch, NumPy, Scikit-learn, and Matplotlib. Installation requires cloning the repository.

cd cnn-bayesian-optimization

- Install the required packages:

pip install -r requirements.txt

Usage- Prepare your dataset and update the data loading script accordingly.
- Adjust the hyperparameters in the config.py file if needed.
- Run the training script:



Example
from model import CNNModel
import torch

# Example usage
image_shape = (32, 32)
kwargs = {
    'kernels': 32,
    'kernel_size': 3,
    'conv_layers': 2,
    'maxpooling': 1,
    'dropout_cnn': 1,
    'dropout_perc_cnn': 0.25,
    'layers': 2,
    'neurons': 128,
    'dropout': 1,
    'dropout_perc': 0.5
}

model = CNNModel(image_shape, **kwargs)
print(model)

Getting Help
If you encounter any issues or have questions about the project, you can open an issue on the GitHub repository.

How to Contribute
To contribute, fork the repository, create a new branch for your feature or bugfix, make changes, and commit them with clear messages.
