# Streamlit-ML-Wine-Predictor
# 🍇 Wine Quality Predictor: ML Deployment with Streamlit

This project demonstrates a complete machine learning pipeline, from data exploration and model training to interactive web application deployment, focusing on predicting the quality of white wines.

## 🚀 Project Overview

The goal of this project is to build a machine learning model that can predict the quality of white wine based on its physicochemical properties and deploy it as an interactive web application using Streamlit. Users can explore the dataset, visualize key features, get real-time predictions, and understand the model's performance.

## 📊 Dataset

The project utilizes the **White Wine Quality Dataset** from the UCI Machine Learning Repository. This dataset contains 11 physicochemical features and a 'quality' score (0-10) for white Vinho Verde wines. For classification, the 'quality' score has been binned into 'Low', 'Medium', and 'High' quality categories.

**Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

## ✨ Features of the Streamlit App

The interactive web application includes several sections accessible via tabs:

* **🏠 Home**: A welcoming page with a brief introduction to the app.
* **🔍 Explore Data**: Provides an overview of the dataset, including raw data, summary statistics, and a correlation heatmap.
* **📊 Visualizations**: Offers interactive plots (histograms/distribution plots and box plots) to explore the relationships between features and wine quality.
* **🎯 Predict Quality**: Allows users to input wine physicochemical parameters via sliders and get a real-time prediction of its quality (Low, Medium, or High). This section also displays the main model's accuracy, a classification report, and a **confusion matrix**.
* **📁 Upload CSV**: Enables users to upload their own CSV file containing wine features to receive batch predictions.
* **📂 Model Info**: Provides details about the deployed machine learning model and allows users to download the trained model file (`model.pkl`).
* **🧪 Model Comparison**: Compares the performance of the trained **Random Forest Classifier** and **Logistic Regression** models on the test set.
* **📜 Feature Guide**: A comprehensive guide explaining each of the wine's physicochemical features.

## 🤖 Machine Learning Models

Two classification algorithms were trained and evaluated:

1.  **Random Forest Classifier**: Selected as the main model due to its superior performance.
2.  **Logistic Regression**: Used for comparison purposes.

Both models were evaluated using a train-test split and **5-fold cross-validation** to ensure robust performance metrics.

## ⚙️ How to Run Locally

To run this Streamlit application on your local machine:

1.  **Clone the Repository**:
    ```bash
    git clone <https://github.com/Lihini0202/Streamlit-ML-Wine-Predictor.git>
    cd <Streamlit-ML-Wine-Predictor>
    ```
2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
    Your app will open in your default web browser.

## ☁️ Deployment

This application is designed for easy deployment on **Streamlit Cloud**. Simply connect your GitHub repository to Streamlit Cloud, and it will automatically build and deploy your application.

