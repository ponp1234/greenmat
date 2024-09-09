

# GreenMat - Sensitive Data Detection Using Regex and Entropy

[![Python application](https://github.com/greenmat007/greenmat/actions/workflows/python-app.yml/badge.svg)](https://github.com/greenmat007/greenmat/actions/workflows/python-app.yml)

**Project Overview**: 
   
GreenMat tool  detect sensitive data in files using a machine learning model trained on regex-based and entropy-based features. The model is capable of identifying sensitive information, such as passwords, tokens, API keys, and numeric patterns (e.g., phone numbers), by processing the content of a file line by line.

## Features
- **Regex-Based Feature Extraction**: Detects predefined patterns using regular expressions.
- **Entropy Calculation**: Measures the randomness of text content to help identify sensitive data like API keys.
- **Machine Learning Model**: A Random Forest classifier trained on combined regex and entropy features.
- **Flexible Regex Patterns**: The regex patterns are stored in a separate JSON file (`regex_patterns.json`), making it easy to update or extend the detection rules.

**Setup and Installation**: 

### Prerequisites

- Python 3.x

### Installation

1. Clone the repository or download the project files.
2. Ensure you have Python 3.x installed.
3. Install the required dependencies using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt

 **License**





