---

# Student AI Assistant Usage Analysis and Prediction

## Overview

This project analyzes student interaction data with an AI assistant and builds multiple machine learning models to predict two key outcomes:

1. **FinalOutcome** – Encoded target representing the result of AI assistant usage.
2. **UsedAgain** – A binary indicator of whether the student would use the assistant again.

The project includes **data exploration, preprocessing, visualization, model training, hyperparameter tuning, and performance evaluation** using several classification algorithms.

---

## Features

* Data exploration and descriptive statistics.
* Visualization of session behavior (histograms, bar charts).
* Data preprocessing (handling missing values, encoding categorical features, dropping irrelevant fields).
* Supervised learning models applied:

  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
  * K-Nearest Neighbors (KNN)
  * Naive Bayes
  * Gradient Boosting
  * XGBoost
* Model evaluation with classification reports, accuracy scores, confusion matrices, and cross-validation.
* Hyperparameter tuning using GridSearchCV for Decision Trees and Random Forests.
* Model comparison framework to evaluate all classifiers consistently.

---

## Dataset

The dataset is assumed to be in CSV format with the filename:

```
ai_assistant_usage_student_life.csv
```

### Example Columns:

* **SessionID** – Unique session identifier (dropped as irrelevant).
* **SessionDate** – Date of the session (dropped as irrelevant).
* **SessionLengthMin** – Session duration in minutes.
* **TotalPrompts** – Number of prompts issued.
* **StudentLevel** – Academic level of the student.
* **Discipline** – Field of study.
* **TaskType** – Type of task performed.
* **SessionLengthCategory** – Categorical grouping of session lengths.
* **UsedAgain** – Binary variable indicating repeat usage.
* **FinalOutcome** – Target label describing session success.

---

## Installation

### Requirements

* Python 3.8+
* Required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## Usage

1. Place the dataset in the working directory.
2. Run the Python script or Jupyter Notebook step by step.
3. The script will:

   * Load and preprocess data.
   * Train multiple classifiers.
   * Evaluate performance metrics.
   * Display best hyperparameters for tuned models.

Example execution (Jupyter Notebook or script):

```bash
python student_ai_assistant_analysis.py
```

---

## Project Workflow

1. **Data Loading & Exploration**

   * Inspect dataset shape, column types, missing values, and summary statistics.
   * Generate frequency counts for categorical variables.

2. **Visualization**

   * Histogram of session lengths.
   * Bar chart of student level distribution.

3. **Preprocessing**

   * Drop irrelevant columns (`SessionID`, `SessionDate`).
   * Encode binary and categorical features using `LabelEncoder` and `OneHotEncoder`.
   * Prepare separate datasets for predicting `FinalOutcome` and `UsedAgain`.

4. **Model Training**

   * Train and evaluate multiple classifiers.
   * Compare performance across models.

5. **Hyperparameter Tuning**

   * Use `GridSearchCV` to optimize Decision Tree and Random Forest parameters.

6. **Evaluation**

   * Generate classification reports (precision, recall, F1-score).
   * Compute confusion matrices.
   * Perform cross-validation for logistic regression.
   * Compare classifiers for predicting `UsedAgain`.

---

## Example Output

* **Classification Report (Decision Tree on FinalOutcome):**
  Shows precision, recall, F1-score, and support for each class.

* **Confusion Matrix (KNN on FinalOutcome):**
  Provides insight into misclassification patterns.

* **Best Hyperparameters (GridSearchCV):**
  Displays tuned model parameters for Decision Trees and Random Forests.

---

## Project Structure

```
project/
│── ai_assistant_usage_student_life.csv   # Dataset  
│── student_ai_assistant_analysis.py      # Main script  
│── README.md                             # Project documentation  
```

---

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with detailed explanations of modifications.

---

Do you want me to **keep this README as a general-purpose guide** or make it more **step-by-step (aligned exactly with your numbered code cells 1–40)**?

