import numpy as np
import matplotlib.pyplot as plt

# параметры варианта 4
a, b, c, d = 0.4, 0.4, 0.08, 0.4
n_in = 6
n_hid = 2
epochs = 2000

# генерация данных
x = np.linspace(0, 20, 200)
y = a * np.cos(b * x) + c * np.sin(d * x)

# делаем выборку по окнам
X = []
Y = []
for i in range(n_in, len(y)):
    X.append(y[i-n_in:i])
    Y.append(y[i])
X = np.array(X)
Y = np.array(Y)

# делим на train/test
n_train = int(0.7 * len(X))
Xtr, Ytr = X[:n_train], Y[:n_train]
Xte, Yte = X[n_train:], Y[n_train:]

# нормировка
mx = Xtr.mean(axis=0)
sx = Xtr.std(axis=0) + 1e-8
my = Ytr.mean()
sy = Ytr.std() + 1e-8

Xtr = (Xtr - mx) / sx
Xte = (Xte - mx) / sx
Ytr = (Ytr - my) / sy
Yte = (Yte - my) / sy

# сигмоида
def sig(z):
    return 1 / (1 + np.exp(-z))

# инициализация весов
np.random.seed(1)
W1 = np.random.randn(n_in, n_hid) * 0.5
b1 = np.zeros(n_hid)
W2 = np.random.randn(n_hid, 1) * 0.5
b2 = np.zeros(1)

# обучение
def train(alpha):
    global W1, b1, W2, b2
    np.random.seed(1)
    W1 = np.random.randn(n_in, n_hid) * 0.5
    b1 = np.zeros(n_hid)
    W2 = np.random.randn(n_hid, 1) * 0.5
    b2 = np.zeros(1)
    errs = []
    for ep in range(epochs):
        # прямой проход
        Z1 = Xtr @ W1 + b1
        A1 = sig(Z1)
        Z2 = A1 @ W2 + b2
        pred = Z2.flatten()
        err = np.mean((pred - Ytr) ** 2)
        errs.append(err)
        # обратный проход
        dZ2 = 2 * (pred - Ytr) / len(Ytr)
        dW2 = A1.T @ dZ2.reshape(-1, 1)
        db2 = dZ2.sum()
        dA1 = dZ2.reshape(-1, 1) @ W2.T
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = Xtr.T @ dZ1
        db1 = dZ1.sum(axis=0)
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
    return errs

# подбираем alpha
alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
best = None
best_err = 1e9
best_losses = None
for al in alphas:
    losses = train(al)
    # ошибка на тесте
    A1 = sig(Xte @ W1 + b1)
    p = (A1 @ W2 + b2).flatten()
    e = np.mean((p - Yte) ** 2)
    print("alpha =", al, " test err =", round(e, 6))
    if e < best_err:
        best_err = e
        best = al
        best_losses = losses

print("лучшая alpha =", best)
losses = train(best)

# прогнозы в исходном масштабе
A1 = sig(Xtr @ W1 + b1)
ptr = (A1 @ W2 + b2).flatten() * sy + my
A1 = sig(Xte @ W1 + b1)
pte = (A1 @ W2 + b2).flatten() * sy + my
Ytr0 = Ytr * sy + my
Yte0 = Yte * sy + my

# Таблица обучения
print("\nОбучение:")
print(f"{'i':>4} | {'эталон':>10} | {'прогноз':>10} | {'отклонение':>12}")
print("-" * 45)
for i in range(len(Ytr0)):
    print(f"{i:>4} | {Ytr0[i]:>10.4f} | {ptr[i]:>10.4f} | {ptr[i] - Ytr0[i]:>12.4f}")

# Таблица теста
print("\nТест:")
print(f"{'i':>4} | {'эталон':>10} | {'прогноз':>10} | {'отклонение':>12}")
print("-" * 45)
for i in range(len(Yte0)):
    print(f"{i:>4} | {Yte0[i]:>10.4f} | {pte[i]:>10.4f} | {pte[i] - Yte0[i]:>12.4f}")

# графики
plt.figure()
plt.plot(x, y, 'k-', label='функция')
plt.plot(x[n_in:n_in+len(ptr)], ptr, 'b.', label='train')
plt.plot(x[n_in+len(ptr):n_in+len(ptr)+len(pte)], pte, 'r.', label='test')
plt.legend()
plt.grid()
plt.title('прогноз')
plt.show()

plt.figure()
plt.plot(losses)
plt.title('ошибка по итерациям')
plt.xlabel('эпоха')
plt.ylabel('MSE')
plt.grid()
plt.show()