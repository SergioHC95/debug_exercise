import numpy as np

def run(steps: int = 200, lr: float = 0.1, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    d = 10
    n_train = d
    n_test = d
    sigma = 0.05

    # Orthonormal design matrices for stable optimization.
    X_train, _ = np.linalg.qr(rng.normal(size=(n_train, d)))
    X_test,  _ = np.linalg.qr(rng.normal(size=(n_test,  d)))

    w_true = rng.normal(size=d)
    y_train = X_train @ w_true + sigma * rng.normal(size=n_train)
    y_test  = X_test  @ w_true + sigma * rng.normal(size=n_test)

    w = rng.normal(size=d)

    for t in range(steps):
        preds = X_train @ w
        err = preds - y_train
        loss = (err @ err) / n_train

        grad = (2 / n_train) * (X_train @ err)
        w = w + lr * grad

        if t % 20 == 0:
            print(f"step {t:4d}  train_loss {loss:.6f}")

    train_loss = np.mean((X_train @ w - y_train) ** 2)
    test_loss  = np.mean((X_test @ w - y_train) ** 2)

    print("final train_loss:", float(train_loss))
    print("final test_loss :", float(test_loss))

