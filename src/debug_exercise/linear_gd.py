import numpy as np


# DO NOT EDIT THIS FUNCTION
def create_data(seed: int = 0):
    rng = np.random.default_rng(seed)

    d = 10
    n_train = 10 * d
    n_test  = 10 * d
    sigma = 0.05

    X_train = rng.normal(size=(n_train, d))
    X_test  = rng.normal(size=(n_test,  d))

    w_true = rng.normal(size=d)

    y_train = X_train @ w_true + sigma * rng.normal(size=n_train)
    y_test  = X_test  @ w_true + sigma * rng.normal(size=n_test)

    return X_train, y_train, X_test, y_test, d, rng


def run_optimization(X_train, y_train, d, rng, lr: float = 0.1, epochs: int = 50, batch_size: int = 10):
    n_train = X_train.shape[0]
    w = rng.normal(size=d)

    for epoch in range(epochs):
        perm = rng.permutation(n_train)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            Xb = X_train[idx]
            yb = y_train[idx]

            err = Xb @ w - yb

            grad = (2 / batch_size) * (Xb @ err)
            w = w + lr * grad

        if epoch % 5 == 0:
            train_loss = np.mean((X_train @ w - y_train) ** 2)
            print(f"epoch {epoch:3d}  train_loss {train_loss:.6f}")

    return w


def evaluate(w, X_train, y_train, X_test, y_test):
    train_loss = np.mean((X_train @ w - y_train) ** 2)
    test_loss  = np.mean((X_test  @ w - y_train) ** 2)

    print("final train_loss:", train_loss)
    print("final test_loss :", test_loss)


def run(epochs: int = 200, lr: float = 0.1, batch_size: int = 10, seed: int = 0) -> None:
    X_train, y_train, X_test, y_test, d, rng = create_data(seed)
    w = run_optimization(X_train, y_train, d, rng, lr, epochs, batch_size)
    evaluate(w, X_train, y_train, X_test, y_test)
