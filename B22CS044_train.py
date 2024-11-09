import sys
import numpy as np

def load(file_path):
    try:
        data = np.loadtxt(file_path, skiprows=1)
        X = data[:, :-1]
        y = data[:, -1]
        return X, y
    except Exception as e:
        print("Error loading data:", str(e))
        sys.exit(1)

def normalize(X):
    try:
        norms = np.linalg.norm(X, axis=0)
        X_norm = X / norms
        return X_norm
    except Exception as e:
        print("Error normalizing data:", str(e))
        sys.exit(1)

def predict(w, x):
    activation = np.dot(w[1:], x) + w[0]
    return 1 if activation >= 0 else 0

def train(X, y, num_feat, epochs=100):
    weights = np.zeros(num_feat + 1) 
    for _ in range(epochs):
        for i in range(len(X)):
            pred = predict(weights, X[i])
            update = (y[i] - pred)
            weights[1:] += update*X[i]
            weights[0] += update 
    return weights

def save(w, filename):
    np.savetxt(filename, w, fmt='%f')
    print("Training done")

def main():
    train_file = sys.argv[1]

    # Load data
    X, y = load(train_file)

    # Normalize the features
    X_norm = normalize(X)

    # Train the perceptron
    weights = train(X_norm, y, X_norm.shape[1])

    # Save weights to file
    save(weights, 'weights.txt')

if __name__ == "__main__":
    main()
