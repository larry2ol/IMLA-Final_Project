Model Card

MODEL DETAILS
Model Description:
The 'CNNModel' is a Convolutional Neural Network (CNN) designed for image classification tasks. The architecture is optimized using Bayesian Optimization to fine-tune hyperparameters such as the number of convolutional layers, kernel size, number of kernels, and dropout rates.

MODEL SPECIFICATIONS
Model Type: Convolutional Neural Network
Model Architecture:
    - Two Convolutional 2D layers with ReLU activation
    - MaxPooling and Dropout layers
    - Adaptive pooling to fixe image size
    - Three Fully connected layers for classification
    - Sigmoid activation for binary classification

INTENDED USES
Primary Intended Uses:
- Image classification tasks in various domains, such as medical imaging and object detection.
Primary Intended Users:
- Junior data scientists and machine learning engineers work on image classification problems.
Out-of-Scope Use Cases:
- Tasks requiring multi-class classification
- Non-image data classification.

APPLICATIONS
Possible Applications:
- Security systems (facial recognition)
 - Medical image analysis
- Autonomous driving (object detection)

ENVIRONMENT
Training Environment: 
-High-performance computing environments with CPUs or GPUs
- Inference Environment: Can be deployed on cloud platforms or edge devices with sufficient computational power.

TRADE-OFFS
- Performance vs. Complexity: Increasing the number of layers and kernels can improve performance but also increases computational complexity and training time.
- Dropout Usage: Helps in regularization but may slow down training.

METRICS AND EVALUATION
Model Performance Measures: Accuracy (Confusion Matrix)
- Decision Thresholds: The default threshold for binary classification is 0.5.

DATASETS AND RESULTS
-Datasets: A publicly available image dataset, AI Generated Images vs Real Images (kaggle.com), is a collection of images of various subjects sourced from web scraping and AI-generated content. 
- Preprocessing: Normalization, data augmentation (e.g., rotation, flipping).
- Results: Between 51 – 60 percent based on the whole dataset. AI-Generated images part of the images yield poor performance under training.

FAIRNESS CRITERIA AND MODEL UPDATES
- Fairness Constraints: Ensure balanced datasets to avoid bias.
- Model Updates: Regular updates based on new data and performance evaluation.

TRAINING ALGORITHMS AND PARAMETERS
- Machine Learning Type: Deep learning
- Modality: Image data
- Training Algorithms: Adam optimizer
- Parameters: Learning rate, batch size, number of epochs, CNN's parameters

ADDITIONAL INFORMATION
Paper or Resource: Refer to standard CNN literature and Bayesian Optimization techniques for more information.
- Citation Details:
Escalante, E. H. J., & Hofmann, K. (2021). Bayesian Optimization is Superior to Random Search for Machine Learning Hyperparameter Tuning: Analysis of the Black-Box Optimization Challenge 2020. In Proceedings of Machine Learning Research (Vol. 133). 
Automated Machine Learning. (2019). In F. Hutter, L. Kotthoff, & J. Vanschoren (Eds.), The Springer Series on Challenges in Machine Learning. Springer International Publishing. https://doi.org/10.1007/978-3-030-05318-5

- Contact: Contact the model developer or maintainer for questions or comments.

ETHICAL CONSIDERATIONS
Caveats and Recommendations: To avoid bias, ensure the model is tested on diverse datasets. Regularly update the model to maintain performance and fairness.
