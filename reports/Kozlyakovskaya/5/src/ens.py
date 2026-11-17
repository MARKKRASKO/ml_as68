import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# Фиксируем случайность, чтобы результаты можно было воспроизвести
torch.manual_seed(42)
np.random.seed(42)

# 1. Генерация исходных данных
x = np.arange(0.0, 100.0 + 1e-9, 0.1, dtype=np.float32)
y = 0.3 * np.cos(0.3 * x) + 0.07 * np.sin(0.3 * x)

# 2. Формирование примеров: 10 предыдущих значений -> следующее значение
window_size = 10
X_all = np.array([y[i-window_size:i] 
                  for i in range(window_size, len(y))], 
                  dtype=np.float32)
target_all = np.array([y[i] 
                       for i in range(window_size, len(y))], 
                       dtype=np.float32).reshape(-1, 1)
target_indices = np.arange(window_size, len(y))

# Первые 80% последовательности для обучения, оставшиеся 20% для тестирования
split_index = int(len(y) * 0.8)
train_mask = target_indices < split_index
test_mask = target_indices >= split_index

X_train = torch.tensor(X_all[train_mask])
y_train = torch.tensor(target_all[train_mask])
X_test = torch.tensor(X_all[test_mask])
y_test = target_all[test_mask]

# 3. Архитектура ИНС: 10 входов -> 4 нейрона Sigmoid -> 1 линейный выход
class ForecastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 4)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        return self.output(x)  # линейная выходная функция

model = ForecastNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Обучение
epochs = 1000
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# 5. Получение результатов
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy().flatten()
    test_pred = model(X_test).numpy().flatten()

train_actual = y_train.numpy().flatten()
test_actual = y_test.flatten()

# Отклонение = эталонное значение - полученное значение
train_table = pd.DataFrame({
    "Эталонное значение": train_actual,
    "Полученное значение": train_pred,
    "Отклонение": train_actual - train_pred
})
test_table = pd.DataFrame({
    "Эталонное значение": test_actual,
    "Полученное значение": test_pred,
    "Отклонение": test_actual - test_pred
})

train_table.to_csv("training_results.csv", index=False, encoding="utf-8-sig")
test_table.to_csv("forecast_results.csv", index=False, encoding="utf-8-sig")

# График функции на участке обучения
train_x = x[target_indices[train_mask]]
plt.figure(figsize=(11, 5))
plt.plot(train_x, train_actual, label="Эталонная функция")
plt.plot(train_x, train_pred, label="Значения ИНС")
plt.title("Прогнозируемая функция на участке обучения")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_plot.png", dpi=160)
plt.show()

# График изменения ошибки по эпохам
plt.figure(figsize=(11, 5))
plt.plot(range(1, epochs + 1), loss_history)
plt.title("Изменение ошибки в зависимости от итерации")
plt.xlabel("Итерация (эпоха)")
plt.ylabel("Среднеквадратичная ошибка (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=160)
plt.show()

# Дополнительный график тестового прогнозирования
test_x = x[target_indices[test_mask]]
plt.figure(figsize=(11, 5))
plt.plot(test_x, test_actual, label="Эталонные значения")
plt.plot(test_x, test_pred, label="Прогноз ИНС")
plt.title("Результаты прогнозирования на тестовом участке")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=160)
plt.show()

print("MSE на обучении:", np.mean((train_actual - train_pred) ** 2))
print("MSE на тестировании:", np.mean((test_actual - test_pred) ** 2))
