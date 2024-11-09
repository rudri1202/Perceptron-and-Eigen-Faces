import sys
import numpy as np

def calc(weights, x):
    activation = np.dot(weights[1:], x) + weights[0]
    return 1 if activation >= 0 else 0

def load_weights(file_path):
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print("Error loading weights:", str(e))
        sys.exit(1)

def normalize_data(X):
    try:
        norms = np.linalg.norm(X, axis=0)
        X_norm = X / norms
        return X_norm
    except Exception as e:
        print("Error normalizing data:", str(e))
        sys.exit(1)

def load_data(file_path):
    try:
        return np.loadtxt(file_path, skiprows=1)
    except Exception as e:
        print("Error loading data:", str(e))
        sys.exit(1)

def predict(weights, data):
    return [calc(weights, x) for x in data]


if __name__ == "__main__":
    test_file = sys.argv[1]
    weights = load_weights('weights.txt')

    test_data = load_data(test_file)
    X = test_data  
    X_norm = normalize_data(X)

    predictions = predict(weights, X_norm)
    print(','.join(map(str, predictions)))
