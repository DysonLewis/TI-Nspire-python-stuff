# TI-Nspire compatible perceptron trainer

# Compute dot product of two lists
def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))

# Train perceptron
def perceptron_train(X, y, mode="once", W=None):
    n = len(X)
    m = len(X[0])
    if W is None:
        W = [0] * (m + 1)  # include bias

    if mode == "once":
        for i in range(n):
            xb = [1] + X[i]  # add bias
            if y[i] * dot(W, xb) <= 0:
                W = [wi + y[i]*xi for wi, xi in zip(W, xb)]
        return W

    elif mode == "until":
        changed = True
        iterations = 0
        while changed:
            changed = False
            for i in range(n):
                xb = [1] + X[i]  # add bias
                if y[i] * dot(W, xb) <= 0:
                    W = [wi + y[i]*xi for wi, xi in zip(W, xb)]
                    changed = True
                    print("Step {}: W = {}".format(iterations + 1, W))
            iterations += 1
        print("Converged after {} iterations".format(iterations))
        return W

# Predict class for new data points
def perceptron_predict(W, X):
    predictions = []
    for x in X:
        xb = [1] + x
        val = dot(W, xb)
        if val > 0:
            predictions.append(1)
        elif val < 0:
            predictions.append(-1)
        else:
            predictions.append(0)
    return predictions


# ------------------------------
# Main interactive section
# ------------------------------

print("Perceptron Trainer")
n = int(input("Enter number of data points: "))
m = int(input("Enter number of features: "))

# Input dataset
X = []
y = []
print("Enter each data point and label (comma-separated, label last).")
for i in range(n):
    vals = list(map(float, input("Point {}: ".format(i + 1)).split(',')))
    X.append(vals[:-1])
    y.append(vals[-1])

mode = input("Run once or until convergence? (once/until): ").strip().lower()
W = perceptron_train(X, y, mode)
print("Final weights:", W)

# Loop for testing after training
while True:
    print("\nEnter a test point (comma-separated):")
    test = list(map(float, input().split(',')))
    pred = perceptron_predict(W, [test])[0]
    print("Predicted class:", int(pred))

    cont = input("Continue training until convergence? (y/n): ").strip().lower()
    if cont == "y":
        W = perceptron_train(X, y, mode="until", W=W)
        print("Updated weights:", W)
    else:
        again = input("Test another point? (y/n): ").strip().lower()
        if again != "y":
            break
