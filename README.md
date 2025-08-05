# Sonar Rock vs Mine Prediction

This is a machine learning web app that predicts whether an object is a Rock or a Mine based on sonar signal data. It uses Logistic Regression and is built using Python, Streamlit, and SQLite.
# Live Demo

 [Click here to try the Live Streamlit App](https://sonar-rock-vs-mine-prediction-numopphpkifovjt2upk2xf.streamlit.app/)
## Features

- Upload CSV files with sonar signal data (60 features).
- Predict whether the object is a Rock or a Mine.
- Display model confidence score.
- Log predictions into a SQLite database.
- View prediction history with timestamp, prediction, and confidence.
- Visualize data and prediction trends with graphs.
- Track model performance over time using test data.

## Technologies Used

- Python 3
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SQLite
- Matplotlib
- Seaborn

## File Structure

sonar-rock-vs-mine-prediction/
├── app.py # Streamlit app file
├── sonar.csv # Dataset
├── sonar_model.pkl # Trained model
├── test_data.csv # Test set to track model performance
├── logger.py # Prediction logging logic
├── prediction_log.db # SQLite database
└── README.md # Project description

## How to Run

1. Clone or download the project.

2. Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
How to Use
Upload a CSV file containing 60 features.

View prediction results and confidence.

Explore prediction history and logged data.

Check graphs to understand prediction distribution and accuracy.

Dataset
The dataset used is the Sonar dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

Author
Shruti Tiwari
GitHub: https://github.com/shrutitac17

License
This project is free to use for learning and personal use.
