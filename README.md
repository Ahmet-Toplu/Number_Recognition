# Number Recognition Program
## Overview
This Number Recognition Program is a machine learning project developed by a university student as part of a learning journey in understanding neural networks and their applications. The program uses a simple neural network to recognize handwritten digits (0-9) and aims to showcase the foundational principles of neural network architecture, training, and inference.

## Features
- **Digit Classification:** Recognizes handwritten numbers from 0 to 9.
- **Neural Network Model:** A custom-built neural network architecture tailored for image-based classification.
- **Training and Evaluation:** Allows training on datasets, as well as real-time testing with user-provided images.
- **Fine-Tuning:** Enables fine-tuning of the model using images captured during testing to improve recognition accuracy on specific inputs.
- **Model Exploration:** Experiment with different network configurations, learning rates, and epochs to observe how each parameter affects the model's performance.

## Technologies Used
- **Python:** The main programming language for the project.
- **TensorFlow:** For building, training, and deploying the neural network.
- **NumPy:** For numerical operations and data manipulation.
- **Matplotlib:** For visualizing data and model performance.
- **Tkinter:** For creating a graphical interface to interact with the program.
- **Pillow:** For handling image operations, such as capturing and processing user-drawn input.
- **JSON:** For configuration and data storage management.

## Installation

**Clone the Repository:**

```python3
git clone https://github.com/your-username/number-recognition.git
cd Number_Recognition
```

**Set Up the Environment:** Make sure you have Python 3.8+ installed, then install dependencies:

```python3
pip install -r requirements.txt
```

**Run the Program:** To start training or testing the model, run:

```python3
python Main.py
```

## Usage

1. Training the Model:
    - Run the training script with sample datasets (e.g., MNIST) to train the neural network on recognizing digits.
    - Adjust hyperparameters in config.json for custom training settings.

2. Fine-Tuning with Testing Images:
    - During testing, users can capture challenging images where the model performance may need improvement. These images can then be added to the training set and used for fine-tuning.
    - To fine-tune the model, store new images in a designated folder (e.g., data/fine_tuning) and adjust the script to include these images in the training data.
    - Fine-tuning can improve recognition for specific inputs, enabling a more adaptable and accurate model.

3. Testing the Model:
    - Load the pre-trained model and test it on sample images or custom input to evaluate its performance.

4. Experimenting:
    - Modify neural network parameters in the code to explore how each change affects the model's accuracy and efficiency.

## Project Structure
- Main.py: Entry point for training and testing the model.
- Recognition.py: Contains the neural network and recognition logic.
- images/: Folder containing images used for training, testing, and fine-tuning.
- models/: Folder for saved models and checkpoints.
- counters.json: JSON file to track or store configuration data, counters, or session information for the model.

## Learning Objectives
**This project is designed to:**
Introduce the basics of neural networks, including layers, activations, and optimization.
Offer hands-on experience with model training, testing, and fine-tuning.
Provide insight into tuning neural networks for improved performance.

## Acknowledgments
**MNIST Dataset:** Used as the primary dataset for digit recognition.
University Course on Machine Learning: Provided the foundation and guidance for this project.
