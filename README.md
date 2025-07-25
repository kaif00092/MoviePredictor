# Movie Success Predictor 🎬

This project uses a machine learning model to predict whether a movie will be a financial "Success" or a "Flop". The prediction is based on features like the movie's budget, popularity, runtime, and genre.

The model is trained on the TMDB 5000 Movie Dataset and uses a **Random Forest Classifier** to make its predictions.

***

## 🎯 Project Goal

The primary goal is to build a classifier that can distinguish between financially successful and unsuccessful movies. The project involves:
1.  **Data Loading & Cleaning:** Loading the dataset and handling missing values and irrelevant data.
2.  **Feature Engineering:** Creating a target variable (`success`) by comparing a movie's revenue to its budget.
3.  **Model Training:** Training a Random Forest Classifier on the prepared data.
4.  **Model Evaluation:** Assessing the model's performance using metrics like accuracy and a classification report.
5.  **Result Visualization:** Creating a confusion matrix to visualize the model's predictions.

***

## 💾 Dataset

This project uses the **TMDB 5000 Movie Dataset** available on Kaggle. You must download the `tmdb_5000_movies.csv` file from the link below and place it in the project folder.

* **Dataset Link:** [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

***

## 🛠️ How to Run

To run this project on your local machine, follow these steps.

### 1. Prerequisites
Make sure you have Python 3 installed on your system.

### 2.  Install Required Libraries
pip install pandas scikit-learn seaborn matplotlib
### 3.  Execute the Script
python predict.py
### 4. 💻 Technologies Used
Python

Pandas (for data manipulation)

Scikit-learn (for machine learning)

Matplotlib & Seaborn (for data visualization
