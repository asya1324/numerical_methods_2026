import csv
import math
import matplotlib.pyplot as plt


def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # пропускаємо заголовок
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y


def form_matrix(x, m):
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xi ** (i + j) for xi in x)
    return A


def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(yi * (xi ** i) for xi, yi in zip(x, y))
    return b


def gauss_solve(A, b):
    n = len(b)
    A_copy = [row[:] for row in A]
    b_copy = b[:]

    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A_copy[i][k]) > abs(A_copy[max_row][k]):
                max_row = i

        A_copy[k], A_copy[max_row] = A_copy[max_row], A_copy[k]
        b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]

        for i in range(k + 1, n):
            if A_copy[k][k] == 0:
                continue
            factor = A_copy[i][k] / A_copy[k][k]
            for j in range(k, n):
                A_copy[i][j] -= factor * A_copy[k][j]
            b_copy[i] -= factor * b_copy[k]

    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A_copy[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b_copy[i] - s) / A_copy[i][i]

    return x_sol


def polynomial(x_val, coef):
    return sum(c * (x_val ** i) for i, c in enumerate(coef))


def variance(y_true, y_approx):
    n = len(y_true)
    s = sum((y_true[i] - y_approx[i]) ** 2 for i in range(n))
    return math.sqrt(s / n)


def calc_error(f_val, phi_val):
    return abs(f_val - phi_val)


def linear_interp(xi, x_data, y_data):
    if xi <= x_data[0]: return y_data[0]
    if xi >= x_data[-1]: return y_data[-1]
    for k in range(len(x_data) - 1):
        if x_data[k] <= xi <= x_data[k + 1]:
            ratio = (xi - x_data[k]) / (x_data[k + 1] - x_data[k])
            return y_data[k] + ratio * (y_data[k + 1] - y_data[k])
    return 0


x, y = read_data('data.csv')
n_points = len(x)

variances = []
coefficients_dict = {}

for m in range(1, 11):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(A, b_vec)
    coefficients_dict[m] = coef

    y_approx = [polynomial(xi, coef) for xi in x]
    var = variance(y, y_approx)
    variances.append(var)
    print(f"Степінь m={m:2d}, Дисперсія: {var:.4f}")

optimal_m = variances.index(min(variances)) + 1
print(f"\nОптимальний степінь многочлена: m = {optimal_m}")
opt_coef = coefficients_dict[optimal_m]

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), variances, 'bo-', linewidth=2)
plt.title("Залежність дисперсії від степені многочлена m")
plt.xlabel("Степінь m")
plt.ylabel("Дисперсія")
plt.grid(True)
plt.show()

plot_x = [min(x) + i * (max(x) - min(x)) / 100 for i in range(101)]
plot_y = [polynomial(xi, opt_coef) for xi in plot_x]

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'ro', label="Фактичні дані")
plt.plot(plot_x, plot_y, 'b-', label=f"Апроксимація (m={optimal_m})")
plt.title("Метод найменших квадратів (Прогноз температури)")
plt.xlabel("Місяць")
plt.ylabel("Температура (°C)")
plt.legend()
plt.grid(True)
plt.show()

x0, xn = min(x), max(x)
n_intervals = 24
h1 = (xn - x0) / (20 * n_intervals)

num_tab_points = int((xn - x0) / h1) + 1
x_tab = [x0 + i * h1 for i in range(num_tab_points)]

plt.figure(figsize=(10, 6))
for m in range(1, 11):
    coef_m = coefficients_dict[m]
    err_tab = [calc_error(linear_interp(xi, x, y), polynomial(xi, coef_m)) for xi in x_tab]
    plt.plot(x_tab, err_tab, label=f"m={m}")

plt.title("Графіки похибки |f(x) - phi(x)| для m=1..10")
plt.xlabel("Місяць")
plt.ylabel("Похибка")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

for future_month in [25, 26, 27]:
    temp_pred = polynomial(future_month, opt_coef)
    print(f"Місяць {future_month}: {temp_pred:.2f} °C")