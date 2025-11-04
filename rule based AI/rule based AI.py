# ==========================================================
# RULE-BASED AI COMPARISON: FORWARD vs BACKWARD CHAINING
# ==========================================================

# This section was developed with assistance from OpenAI’s ChatGPT (2025).ChatGPT was used to help structure the code logic, optimise readability

import pandas as pd
from sklearn.metrics import classification_report
from google.colab import files # This code was designed to run in Google Colab.

# Load dataset (uploaded via Google Colab file picker)
uploaded = files.upload()
df = pd.read_csv("training_data.csv")

# Keep only relevant columns
df = df[['Age', 'Job', 'Housing', 'Saving accounts', 'Checking account',
         'Credit amount', 'Duration', 'Purpose', 'Risk']]

# ----------------------------------------------------------
# BACKWARD CHAINING FUNCTION
# ----------------------------------------------------------
# Implements a goal-driven reasoning approach.
def backward_chaining_risk(row):
    hypothesis = "bad"
    if hypothesis == "bad":
        # Rule 1: Missing/none checking account, high credit amount, long duration
        if (row['Checking account'] in ['NaN', 'none'] and
            row['Credit amount'] > 4000 and
            row['Duration'] > 24):
            return "bad"
              
        # Rule 2: Low savings, renting, and long duration
        if (row['Saving accounts'] == 'little' and
            row['Housing'] == 'rent' and
            row['Duration'] > 30):
            return "bad"
              
        # Rule 3: Younger applicants borrowing for car or electronics
        if (row['Age'] < 25 and
            row['Purpose'] in ['radio/TV', 'car']):
            return "bad"
              
    # If no rule supports the hypothesis, classify as "good"
    return "good"

# ----------------------------------------------------------
# FORWARD CHAINING FUNCTION
# ----------------------------------------------------------
# Implements a data-driven reasoning approach.
def forward_chaining_risk(row):
    # Rule 1
    if (row['Checking account'] in ['NaN', 'none'] and
        row['Credit amount'] > 4000 and
        row['Duration'] > 24):
        return "bad"
    # Rule 2
    if (row['Saving accounts'] == 'little' and
        row['Housing'] == 'rent' and
        row['Duration'] > 30):
        return "bad"
    # Rule 3
    if (row['Age'] < 25 and
        row['Purpose'] in ['radio/TV', 'car']):
        return "bad"
    # Default classification if no rule is triggered
    return "good"

# ----------------------------------------------------------
# APPLY BOTH SYSTEMS
# ----------------------------------------------------------
# Each function is applied to every record in the dataset. Results are stored in new columns for later comparison.
df['Backward_Predicted_Risk'] = df.apply(backward_chaining_risk, axis=1)
df['Forward_Predicted_Risk'] = df.apply(forward_chaining_risk, axis=1)

# ----------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------
# Evaluation code structure refined with assistance from OpenAI’s ChatGPT (2025).
# Uses scikit-learn’s classification_report to compute standard performance metrics.
print("Backward Chaining Credit Risk Report:\n")
print(classification_report(df['Risk'], df['Backward_Predicted_Risk'])) 

print("Forward Chaining Credit Risk Report:\n")
print(classification_report(df['Risk'], df['Forward_Predicted_Risk']))
