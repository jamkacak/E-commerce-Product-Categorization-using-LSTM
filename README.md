# E-commerce Product Categorization using LSTM

This project focuses on the categorization of e-commerce products into four predefined categories: "Electronics", "Household", "Books", and "Clothing & Accessories". The goal is to use machine learning to automate the categorization process, which is traditionally done manually, to save time and resources. The project uses a Long Short-Term Memory (LSTM) network to classify unseen product data with high accuracy.

## Project Overview

E-commerce platforms generate vast amounts of textual data that contain valuable insights for businesses. By categorizing products into different segments, companies can streamline their operations and focus on specific types of products. However, the manual classification of products is time-consuming, and this project aims to solve this problem using machine learning.

In this project, an LSTM model will be developed to categorize e-commerce products into one of the four categories: 
- Electronics
- Household
- Books
- Clothing & Accessories

The model will be trained using the dataset obtained from Kaggle and evaluated based on its accuracy and F1 score.

## Dataset

The dataset used in this project can be obtained from Kaggle:

- **Dataset URL**: [E-commerce Text Classification Dataset](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-textclassification)

The dataset consists of text descriptions of products, which will be used to train the LSTM model for categorization.

## Project Requirements

### 1. Data Processing
- Download the dataset and read it using **Pandas** to preprocess and prepare the data for training.

### 2. Model Development
- Develop a machine learning model using **LSTM** (Long Short-Term Memory) to classify products into the specified categories.
- Achieve an accuracy of more than **85%** and an **F1 score of more than 0.7**.

### 3. TensorFlow
- Use the **TensorFlow** library for developing and training the model.

### 4. TensorBoard
- Visualize the training process using **TensorBoard**.

### 5. Model & Tokenizer Saving
- Save the trained model in **.h5** format in a folder named **saved_models**.
- Save the tokenizer in **.json** format in the **saved_models** folder.

### 6. Performance Evaluation
- The model’s performance should be evaluated using accuracy and F1 score metrics.

## Deliverables

The following files should be uploaded to both GitHub and LMS:

1. **Training Script**: A Python script containing the code to train the model.
2. **Saved Model**: The trained model in **.h5** format and any scaler files (if used) in **.pkl** format.
3. **TensorBoard Visualization**: A screenshot of the training process visualized using TensorBoard.
4. **Model Architecture Screenshot**: A screenshot of the model’s architecture in **.png** format.
5. **Performance Evaluation**: A screenshot showing the model's performance metrics (accuracy, F1 score, etc.).
6. **GitHub URL**: A text file containing the URL of the GitHub repository with the completed project.
7. **Data Citation**: Credit the source of the dataset in the GitHub repository.

### Folder Structure
- **saved_models/**: Folder containing the model in **.h5** format and tokenizer in **.json** format.
- **scripts/**: Folder containing the Python training script.
- **images/**: Folder containing all images (e.g., model architecture, performance metrics, TensorBoard graphs).
- **data/**: Folder containing the dataset.

## Instructions

1. Download the dataset from Kaggle.
2. Preprocess the data using **Pandas** to prepare it for training.
3. Build an LSTM-based model using **TensorFlow**.
4. Train the model and monitor the training process using **TensorBoard**.
5. Save the trained model in the specified format and structure.
6. Submit the project by zipping all the required files and uploading them to GitHub and LMS by the specified deadline.

## Credits

- **Dataset Source**: [E-commerce Text Classification Dataset on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-textclassification)

Good luck with the project!