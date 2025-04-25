# TRAFFICGAN Project

This project explores the use of Generative Adversarial Networks (GANs) to generate synthetic images of German traffic signs. The primary dataset used is the German Traffic Sign Recognition Benchmark (GTSRB). Various GAN architectures were experimented with, culminating in an LSGAN (Least Squares GAN) implementation.

## Prerequisites

*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   Standard scientific Python libraries (NumPy, Matplotlib, etc.)
*   A deep learning framework (likely TensorFlow/Keras, based on typical GAN implementations)
*   Kaggle account (to download the dataset)

## `gangen` Library

This project includes a custom library named `gangen` located in the `gangen/` directory. This library encapsulates:

*   Implementations of the different GAN model architectures tried (e.g., DCGAN, CGAN, LSGAN).
*   Utility functions for data preprocessing and other common tasks within the project.

## Setup Instructions

1.  **Download the Dataset:**
    *   Obtain the German Traffic Sign Recognition Benchmark (GTSRB) dataset from Kaggle:
    *   [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
    *   Download and extract the dataset to a location on your machine (let's call this the `<original_dataset_path>`).

2.  **Preprocess the Data:**
    *   Open the `a2_preprocessing.ipynb` notebook.
    *   Inside the notebook, find the variable `data_dir` (or similar) and set its value to your `<original_dataset_path>`.
    *   Find the variable `target_dir` (or similar) and set its value to the desired path where the preprocessed data should be saved (let's call this the `<preprocessed_dataset_path>`).
    *   Run all cells in the `a2_preprocessing.ipynb` notebook to execute the preprocessing steps. This will create the necessary dataset format in your specified `<preprocessed_dataset_path>`.

## Running the Latest Model (LSGAN with MSE)

1.  **Execute the Training Notebook:**
    *   Open the `e1_lsgan_mse.ipynb` notebook.
    *   Inside the notebook, find the variable `data_dir` (or similar, likely representing the input data path for training) and set its value to your `<preprocessed_dataset_path>` (the output directory from the preprocessing step).
    *   Run the cells in the notebook to train the LSGAN model and generate synthetic traffic sign images.
