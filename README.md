# Multi-Modal Biometric Authentication System

This repository contains the code and files for the multi-modal biometric authentication system as described in our research paper. The system integrates face, voice, and signature data to enhance security in biometric authentication.

## Features
- **Multi-modal Fusion:** Combines facial, vocal, and signature biometric modalities.
- **CNN and RNN Layers:** Utilizes Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for spatial and temporal feature extraction.
- **PCA and GBM:** Principal Component Analysis (PCA) for dimensionality reduction and Gradient Boosting Machines (GBM) for classification.
  
## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Datasets](#datasets)
5. [Results](#results)
6. [Citing](#citing)

## Installation

### Prerequisites
- Python 3.7 or later
- The required packages can be installed using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

### Additional Dependencies
- TensorFlow or PyTorch (depending on the deep learning library used).
- scikit-learn for machine learning utilities.
- matplotlib for generating graphs and plots.
  
## Usage

### Running the Model
To run the multi-modal biometric system, execute the following command in your terminal:
```bash
python main.py
