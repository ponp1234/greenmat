import re
import numpy as np
from scipy.stats import entropy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to read regex patterns from a file
def read_regex_patterns(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        patterns = [line.strip() for line in file if line.strip()]
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

# Process a single line and extract features
def process_line(line, regex_patterns):
    # Preprocess text: lowercase, remove non-alphanumeric characters
    line = re.sub(r'\W+', ' ', line.lower())
    # Extract regex-based features
    regex_features = regex_match_features(line, regex_patterns)
    # Extract entropy-based feature
    entropy_feature = [calculate_entropy(line)]
    # Combine regex and entropy features into a single feature set
    return np.hstack((regex_features, entropy_feature))

# Read regex patterns from the external file
regex_file_path = 'regex_patterns.txt'  # Path to the file containing regex patterns
regex_patterns = read_regex_patterns(regex_file_path)

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

# Optionally, normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a Random Forest model (or any other classic machine learning model)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# File to search for sensitive data
file_path = 'path_to_your_file.txt'  # Replace with the path to your file

# Process the file line by line and check for sensitive data
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    for line_number, line in enumerate(file, 1):
        # Extract features for the current line
        line_features = process_line(line, regex_patterns)
        # Scale the features using the same scaler as used during training
        line_features_scaled = scaler.transform([line_features])
        # Predict whether the line contains sensitive data
        line_prediction = model.predict(line_features_scaled)
        # Print the line number and the line if sensitive data is detected
        if line_prediction[0] == 1:
            print(f"Sensitive data detected on line {line_number}: {line.strip()}")
