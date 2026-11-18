import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

# ============================================================
# 1. Параметры варианта 3
# ============================================================
A, B, C, D = 0.3, 0.3, 0.07, 0.3
N_IN, N_HID = 10, 4
HORIZON = 20
EPOCHS_SEARCH = 1000
EPOCHS_FINAL = 5000

# Фиксируем seed для воспроизводимости
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. Генерация данных
# ============================================================
x = np.arange(0.0, 20.0 + 1e-9, 0.1, dtype=np.float64)
y = A * np.cos(B * x) + C * np.sin(D * x)

# Нормировка min-max в [0, 1]
y_min, y_max = float(y.min()), float(y.max())
ys = (y - y_min) / (y_max - y_min)

def denorm(v):
    return v * (y_max - y_min) + y_min

# ============================================================
# 3. Формирование окон (скользящее окно)
# ============================================================
X_list, Y_list = [], []
for i in range(len(ys) - N_IN):
    X_list.append(ys[i:i + N_IN])
    Y_list.append(ys[i + N_IN])

X = np.asarray(X_list, dtype=np.float32)
Y = np.asarray(Y_list, dtype=np.float32).reshape(-1, 1)

# 80/20
n_train = int(0.8 * len(X))
Xtr, Ytr = X[:n_train], Y[:n_train]
Xte, Yte = X[n_train:], Y[n_train:]

print(f"Всего окон: {len(X)} | обучающих: {len(Xtr)} | тестовых: {len(Xte)}")

# ============================================================
# 4. Архитектура 10-4-1: сигмоида -> линейный выход
# ============================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_HID)
        self.fc2 = nn.Linear(N_HID, 1)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        return self.fc2(h)

# ============================================================
# 5. Обучение
# ============================================================
def train(alpha, X, Y, epochs, seed=SEED):
    torch.manual_seed(seed)
    model = Net()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses

# ============================================================
# 6. Подбор alpha
# ============================================================
results = []
for a in [0.01, 0.05, 0.1, 0.3, 0.5]:
    _, L = train(a, Xtr, Ytr, epochs=EPOCHS_SEARCH)
    results.append((a, L[-1]))
    print(f"alpha={a:<5} MSE={L[-1]:.6f}")

best_alpha = min(results, key=lambda t: t[1])[0]
print(f"\nЛучшее alpha: {best_alpha}")

# ============================================================
# 7. Финальное обучение
# ============================================================
model, losses = train(best_alpha, Xtr, Ytr, epochs=EPOCHS_FINAL)
model.eval()

# ============================================================
# 8. Результаты на обучении
# ============================================================
with torch.no_grad():
    ptr = denorm(model(torch.from_numpy(Xtr)).numpy().ravel())
    ttr = denorm(Ytr.ravel())

train_df = pd.DataFrame({
    "Эталон": ttr,
    "Получено": ptr,
    "Отклонение": ptr - ttr,
})
print("\nОбучение, первые 10 строк:")
print(train_df.head(10).to_string(index=False))
train_df.to_csv("train.csv", index=False)

rmse_tr = float(np.sqrt(np.mean((ptr - ttr) ** 2)))
mae_tr = float(np.mean(np.abs(ptr - ttr)))
print(f"\nОбучение: RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}")

# ============================================================
# 9. Результаты на тесте
# ============================================================
with torch.no_grad():
    pte = denorm(model(torch.from_numpy(Xte)).numpy().ravel())
    tte = denorm(Yte.ravel())

test_df = pd.DataFrame({
    "Эталон": tte,
    "Получено": pte,
    "Отклонение": pte - tte,
})
print("\nТест, первые 10 строк:")
print(test_df.head(10).to_string(index=False))
test_df.to_csv("test.csv", index=False)

rmse_te = float(np.sqrt(np.mean((pte - tte) ** 2)))
mae_te = float(np.mean(np.abs(pte - tte)))
print(f"\nТест: RMSE={rmse_te:.6f}, MAE={mae_te:.6f}")

# ============================================================
# 10. График ошибки обучения
# ============================================================
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Итерация")
plt.ylabel("MSE")
plt.title(f"Ошибка обучения, alpha={best_alpha}")
plt.grid()
plt.tight_layout()
plt.savefig("loss.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 11. График прогноза на участке обучения
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(ttr, label="Эталон")
plt.plot(ptr, label="ИНС")
plt.xlabel("Номер окна")
plt.ylabel("y")
plt.title("Прогноз на участке обучения")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("train.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 12. Многошаговый прогноз (рекуррентный)
# ============================================================
window = ys[n_train:n_train + N_IN].astype(np.float32).copy()
fc = []
with torch.no_grad():
    for _ in range(HORIZON):
        inp = torch.from_numpy(window.reshape(1, -1))
        v = float(model(inp).item())
        fc.append(denorm(v))
        window = np.append(window[1:], np.float32(v))

true_fc = y[n_train + N_IN : n_train + N_IN + HORIZON]

fc_df = pd.DataFrame({
    "Эталон": true_fc,
    "Получено": fc,
    "Отклонение": np.asarray(fc) - true_fc,
})
print("\nМногошаговый прогноз:")
print(fc_df.to_string(index=False))
fc_df.to_csv("forecast.csv", index=False)

rmse_fc = float(np.sqrt(np.mean((np.asarray(fc) - true_fc) ** 2)))
mae_fc = float(np.mean(np.abs(np.asarray(fc) - true_fc)))
print(f"\nПрогноз: RMSE={rmse_fc:.6f}, MAE={mae_fc:.6f}")

plt.figure(figsize=(10, 4))
plt.plot(true_fc, marker="o", label="Эталон")
plt.plot(fc, marker="s", label="ИНС")
plt.xlabel("Шаг")
plt.ylabel("y")
plt.title("Многошаговый прогноз")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("forecast.png", dpi=150)
plt.show()
plt.close()