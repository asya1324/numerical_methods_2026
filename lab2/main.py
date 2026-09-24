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


def omega(x_val, x_nodes, k):
    res = 1.0
    for i in range(k + 1):
        res *= (x_val - x_nodes[i])
    return res


def divided_differences(x_nodes, y_nodes):
    n = len(y_nodes)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y_nodes[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_nodes[i + j] - x_nodes[i])

    return [table[0][j] for j in range(n)]


def newton_polynomial(x_val, x_nodes, div_diffs):
    n = len(x_nodes)
    result = div_diffs[0]  # f_0
    for k in range(1, n):
        # N_n(x) = f_0 + cyma( w_{k-1}(x) * f(x_0, ..., x_k) )
        result += omega(x_val, x_nodes, k - 1) * div_diffs[k]
    return result


def calc_error(y_true, y_approx):
    return abs(y_true - y_approx)


def finite_differences(y_nodes):
    n = len(y_nodes)
    diffs = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diffs[i][0] = y_nodes[i]

    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i + 1][j - 1] - diffs[i][j - 1]
    return [diffs[0][j] for j in range(n)]


def factorial_polynomial(t, y_nodes, fin_diffs):
    n = len(y_nodes)
    result = fin_diffs[0]
    for k in range(1, n):
        t_fac = 1.0
        for i in range(k):
            t_fac *= (t - i)
        result += (t_fac / math.factorial(k)) * fin_diffs[k]
    return result

try:
    x_data, y_data = read_data('data.csv')
except FileNotFoundError:
    print("Файл data.csv не знайдено, використовуються стандартні дані Варіанта 1.")
    x_data = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    y_data = [3.0, 5.0, 11.0, 28.0, 85.0]

n_pred = 6000.0

div_diffs = divided_differences(x_data, y_data)
time_newton = newton_polynomial(n_pred, x_data, div_diffs)
print(f"1) Прогноз поліномом Ньютона для n=6000: {time_newton:.2f} мс")

fin_diffs = finite_differences(y_data)
t_pred = math.log2(n_pred / 1000.0)
time_factorial = factorial_polynomial(t_pred, y_data, fin_diffs)
print(f"2) Прогноз Факторіальним многочленом для n=6000: {time_factorial:.2f} мс")

plot_x = [1000 + i * (16000 - 1000) / 100 for i in range(101)]
plot_y = [newton_polynomial(xi, x_data, div_diffs) for xi in plot_x]
plt.figure(figsize=(8, 5))
plt.plot(x_data, y_data, 'ro', label="Експериментальні дані")
plt.plot(plot_x, plot_y, 'b-', label="Інтерполяція Ньютона")
plt.plot(n_pred, time_newton, 'g*', markersize=12, label=f"Прогноз (n={int(n_pred)})")
plt.title("Варіант 1: Залежність часу від розміру даних")
plt.xlabel("n (розмір даних)")
plt.ylabel("t (час, мс)")
plt.legend()
plt.grid(True)
plt.show()


def f_runge(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


a, b = -1.0, 1.0
node_counts = [5, 10, 20]

for n in node_counts:
    print(f"\nАналіз для n={n} вузлів:")

    h_nodes = (b - a) / (n - 1)
    x_nodes = [a + i * h_nodes for i in range(n)]
    y_nodes = [f_runge(xi) for xi in x_nodes]

    div_diffs_runge = divided_differences(x_nodes, y_nodes)

    h_tab = (b - a) / (20 * n)
    num_tab_points = int((b - a) / h_tab) + 1

    x_tab = [a + i * h_tab for i in range(num_tab_points)]
    f_tab = [f_runge(xi) for xi in x_tab]
    N_tab = [newton_polynomial(xi, x_nodes, div_diffs_runge) for xi in x_tab]
    err_tab = [calc_error(f_tab[i], N_tab[i]) for i in range(num_tab_points)]
    omega_tab = [omega(xi, x_nodes, n - 1) for xi in x_tab]

    print(f"Максимальна похибка інтерполяції (n={n}): {max(err_tab):.6f}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Дослідження інтерполяції для n={n} вузлів")

    axs[0].plot(x_tab, f_tab, 'k-', label="f(x)")
    axs[0].plot(x_tab, N_tab, 'b--', label="N_n(x)")
    axs[0].plot(x_nodes, y_nodes, 'ro', label="Вузли")
    axs[0].set_title("Функція та поліном Ньютона")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylim(-0.5, 1.5)

    axs[1].plot(x_tab, err_tab, 'r-')
    axs[1].set_title("Похибка e(x) = |f(x) - N_n(x)|")
    axs[1].grid(True)

    axs[2].plot(x_tab, omega_tab, 'g-')
    axs[2].set_title("Функція w_n(x)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()