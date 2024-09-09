import re
import numpy as np
import joblib
from scipy.stats import entropy
from collections import Counter

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

# Function to process a single line and extract features
def process_line(line, regex_patterns):
    # Preprocess text: lowercase, remove non-alphanumeric characters
    line = re.sub(r'\W+', ' ', line.lower())
    # Extract regex-based features
    regex_features = regex_match_features(line, regex_patterns)
    # Extract entropy-based feature
    entropy_feature = [calculate_entropy(line)]
    # Combine regex and entropy features into a single feature set
    return np.hstack((regex_features, entropy_feature))

# Load the saved model and scaler
model = joblib.load('sensitive_data_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load regex patterns from the text file
regex_txt_file = 'regex-config.txt'  # Path to the text file containing regex patterns
regex_patterns = read_regex_patterns(regex_txt_file)

# File to search for sensitive data
file_path = 'path_to_your_file.txt'  # Replace with the path to your file

# Process the file line by line and check for sensitive data
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    for line_number, line in enumerate(file, 1):
        # Extract features for the current line
        line_features = process_line(line, regex_patterns)
        # Scale the features using the same scaler used during training
        line_features_scaled = scaler.transform([line_features])
        # Predict whether the line contains sensitive data
        line_prediction = model.predict(line_features_scaled)
        # Print the line number and the line if sensitive data is detected
        if line_prediction[0] == 1:
            print(f"Sensitive data detected on line {line_number}: {line.strip()}")
