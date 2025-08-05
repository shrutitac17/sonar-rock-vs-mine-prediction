import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from datetime import datetime

# Load dataset
df = pd.read_csv("sonar.csv", header=None)

# Prepare data
X = df.iloc[:, :-1]
y = df.iloc[:, -1].map({'R': 0, 'M': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save model
with open("sonar_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Log performance
log_file = "model_performance_log.csv"
log_entry = f"{datetime.now()},{accuracy:.4f}\n"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,accuracy\n")
with open(log_file, "a") as f:
    f.write(log_entry)

import pandas as pd
from datetime import datetime

# Accuracy already calculated earlier
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
new_row = pd.DataFrame([[timestamp, accuracy]], columns=["timestamp", "accuracy"])

# Append to performance_log.csv
new_row.to_csv("performance_log.csv", mode='a', header=False, index=False)
