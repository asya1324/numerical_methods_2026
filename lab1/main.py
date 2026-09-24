import requests
import urllib3
import math
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url, verify=False)
data = response.json()["results"]

lats = [point["latitude"] for point in data]
lons = [point["longitude"] for point in data]
elevs = [point["elevation"] for point in data]
n_points = len(data)

with open("tabulation.txt", "w") as file:
    file.write("Latitude\tLongitude\tElevation (m)\n")
    for i in range(n_points):
        file.write(f"{lats[i]}\t{lons[i]}\t{elevs[i]}\n")
print("Дані успішно збережено у tabulation.txt")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


distances = [0.0]
for i in range(1, n_points):
    d = haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
    distances.append(distances[-1] + d)

x_full = distances
y_full = elevs


def build_cubic_spline(x, y, print_coeffs=False):
    n = len(x)
    h = [x[i] - x[i - 1] for i in range(1, n)]

    alpha = [0] * n
    beta = [0] * n
    gamma = [0] * n
    delta = [0] * n

    beta[0] = 1
    beta[n - 1] = 1

    for i in range(1, n - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    if print_coeffs:
        print("\n=== Коефіцієнти системи рівнянь ===")
        for i in range(1, n - 1):
            print(f"i={i}: alpha={alpha[i]:.2f}, beta={beta[i]:.2f}, gamma={gamma[i]:.2f}, delta={delta[i]:.2f}")

    A = [0] * n
    B = [0] * n
    for i in range(1, n - 1):
        denominator = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denominator

    c = [0] * n
    for i in range(n - 2, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    if print_coeffs:
        print("\n=== Коефіцієнти c_i (після прогонки) ===")
        for i in range(n):
            print(f"c[{i}] = {c[i]:.4f}")

    a = [0] * (n - 1)
    b = [0] * (n - 1)
    d = [0] * (n - 1)

    for i in range(n - 1):
        a[i] = y[i]
        b[i] = (y[i + 1] - y[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    if print_coeffs:
        print("\n=== Коефіцієнти a_i, b_i, d_i ===")
        for i in range(n - 1):
            print(f"Інтервал {i}: a={a[i]:.2f}, b={b[i]:.2f}, c={c[i]:.4f}, d={d[i]:.6f}")

    def evaluate(xi):
        for i in range(n - 1):
            if x[i] <= xi <= x[i + 1] or i == n - 2:
                dx = xi - x[i]
                return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
        return 0

    return evaluate


spline_full = build_cubic_spline(x_full, y_full, print_coeffs=True)


def get_subset(k):
    step = (len(x_full) - 1) / (k - 1)
    indices = [int(round(i * step)) for i in range(k)]
    return [x_full[i] for i in indices], [y_full[i] for i in indices]


x_10, y_10 = get_subset(10)
x_15, y_15 = get_subset(15)
x_20, y_20 = get_subset(20)

spline_10 = build_cubic_spline(x_10, y_10)
spline_15 = build_cubic_spline(x_15, y_15)
spline_20 = build_cubic_spline(x_20, y_20)

plot_x = [x_full[0] + i * (x_full[-1] - x_full[0]) / 500 for i in range(501)]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(x_full, y_full, 'ko', label="GPS вузли (21)")
plt.plot(plot_x, [spline_full(xi) for xi in plot_x], 'k-', linewidth=2, label="Еталон (21 вузол)")
plt.plot(plot_x, [spline_10(xi) for xi in plot_x], '--', label="10 вузлів")
plt.plot(plot_x, [spline_15(xi) for xi in plot_x], '-.', label="15 вузлів")
plt.plot(plot_x, [spline_20(xi) for xi in plot_x], ':', label="20 вузлів")
plt.title("Інтерполяція висоти маршруту (Заросляк-Говерла)")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(plot_x, [abs(spline_full(xi) - spline_10(xi)) for xi in plot_x], label="Похибка (10 вузлів)")
plt.plot(plot_x, [abs(spline_full(xi) - spline_15(xi)) for xi in plot_x], label="Похибка (15 вузлів)")
plt.plot(plot_x, [abs(spline_full(xi) - spline_20(xi)) for xi in plot_x], label="Похибка (20 вузлів)")
plt.title("Похибка апроксимації |y - y_набл|")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Похибка (м)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()