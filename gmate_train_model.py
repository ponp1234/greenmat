import re
import numpy as np
from scipy.stats import entropy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Function to read regex patterns from a text file
def read_regex_patterns(txt_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        patterns = [line.strip() for line in file if line.strip()]  # Ignore empty lines
    return patterns

# Function to extract regex match features
def regex_match_features(text, regex_patterns):
    features = []
    for pattern in regex_patterns:
        features.append(int(bool(re.search(pattern, text))))
    return features

# Function to calculate entropy of a string based on word frequency
def calculate_entropy(text):
    words = text.split()
    word_counts = Counter(words)
    probabilities = np.array(list(word_counts.values())) / len(words)
    return entropy(probabilities)

# Read regex patterns from the text file
regex_txt_file = 'regex_patterns.txt'  # Path to the text file containing regex patterns
regex_patterns = read_regex_patterns(regex_txt_file)

# Example dataset for training
data = [
    "My password is password123", 
    "The token is abcd1234xyz",
    "Here is some normal text with no sensitive data",
    "Call 1234567890 for more details",
    "The secret API key is 123456789abcdefg"
]
labels = [1, 1, 0, 0, 1]  # Binary labels: 1 = contains sensitive data, 0 = no sensitive data

# Extract features from the training dataset
regex_features = [regex_match_features(text, regex_patterns) for text in data]
entropy_features = [[calculate_entropy(text)] for text in data]
X = np.hstack((np.array(regex_features), np.array(entropy_features)))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler to disk
joblib.dump(model, 'sensitive_data_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
