# Fuel Blend Properties Prediction

This repository contains a machine learning project focused on predicting the properties of complex fuel blends based on their constituent components and proportions. The goal is to develop accurate predictive models that can estimate 10 key properties of blended fuels, enabling faster development and optimization of sustainable fuel formulations.

## 🌱 Project Overview

The formulation of optimal fuel blends requires balancing multiple constraints:

- Ensuring compliance with **safety and performance standards**
- Enhancing **environmental sustainability**
- Maintaining **economic viability**

This project explores how data-driven modeling can help predict the behavior of fuel blends without the need for costly and time-consuming physical experiments.

By leveraging machine learning techniques, we aim to:
- Rapidly evaluate thousands of potential blend combinations
- Identify optimal fuel recipes that maximize sustainability
- Accelerate the development cycle of new fuel formulations
- Enable real-time optimization in production environments

## 📊 Dataset Description

### Files Included

| File | Description |
|------|-------------|
| `train.csv` | Training data containing blend composition, component properties, and target blend properties |
| `test.csv` | Test data with only input features; model must predict the 10 blend properties |
| `sample_submission.csv` | Template for submission format |

### Column Groups in `train.csv`

1. **Blend Composition (first 5 columns):** Volume percentages of each of the 5 base components.
2. **Component Properties (next 50 columns):** Properties of each component batch (e.g., Component1_Property1).
3. **Final Blend Properties - Targets (last 10 columns):** Target properties to predict (e.g., BlendProperty1).

## 📈 Evaluation Metric

The evaluation metric used is the **Mean Absolute Percentage Error (MAPE)**, defined as:

$$
\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100
$$

Where:
- $y_t$: Actual value
- $\hat{y}_t$: Predicted value

For reporting purposes, scores are normalized using the formula:

$$
\text{Score} = \max(0, 100 - \left(\frac{\text{cost}}{\text{reference cost}}\right) \times 100)
$$

## ⚙️ Solution Approach

Our approach includes the following steps:

1. **Exploratory Data Analysis (EDA):** Understand feature distributions, missing values, and correlations.
2. **Feature Engineering:** Create meaningful features like weighted averages, interaction terms, and statistical aggregations.
3. **Model Selection & Training:** Train robust regression models (e.g., XGBoost, LightGBM, CatBoost) using cross-validation.
4. **Hyperparameter Tuning:** Optimize model parameters for improved accuracy.
5. **Prediction Generation:** Use trained models to generate predictions on test data.
6. **Submission Formatting:** Prepare output in the required format (`ID`, `BlendProperty1` to `BlendProperty10`).

## 🧠 Technologies Used

- **Python** (3.10.18)
- **Pandas**, **NumPy** – For data manipulation
- **Scikit-learn**, **XGBoost**, **LightGBM**, **CatBoost** – For modeling
- **Matplotlib**, **Seaborn** – For visualization
- **Jupyter Notebooks** – For exploratory analysis

## 📁 Repository Structure

```
fuel-blend-properties-prediction/
│
├── data/                  # Raw datasets (train.csv, test.csv)
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── src/                   # Python scripts for preprocessing, training, prediction
├── models/                # Trained models (saved as .joblib/.pkl)
├── submissions/           # Submission files (including sample_submission.csv)
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── .gitignore             # Files to exclude from version control
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fuel-blend-properties-prediction.git  
   cd fuel-blend-properties-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Perform exploratory data analysis:
   ```bash
   jupyter notebook
   # Open notebooks/exploratory_data_analysis.ipynb
   ```

4. Train the model:
   ```bash
   python src/model_training.py
   ```

5. Generate predictions and create submission:
   ```bash
   python src/predictions.py
   ```

## 🏆 Results

- **Test Set MAPE**: [Insert result here]
- **Normalized Score**: [Insert score here]

*(Note: Scores will depend on the reference cost and dataset used)*


## 👥 Contributors

- [Vipin Kumar](github.com/krvipin15)

## 📄 License

This project is licensed under the **MIT License**.
