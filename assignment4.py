"""Solutions to assignment 4: Newton's method (NumPy and SciPy)."""

import numpy as np
from scipy import linalg


def newton(x, f, Df, cond, history=None, max_iter=100):
    """Solve J(x) * step = f(x), then update x <- x - step."""
    x = np.array(x, dtype=float, copy=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for i in range(max_iter):
            DFinv_Fx = linalg.solve(Df(x), f(x))
            x = x - DFinv_Fx
            if not np.all(np.isfinite(x)):
                raise FloatingPointError("Non-finite Newton iterate")
            if history is not None:
                history.append((x.copy(), DFinv_Fx.copy()))
            if cond(i, x, DFinv_Fx):
                return x
    raise RuntimeError("Newton method did not converge within max_iter")


def f1(x):
    return np.array([x[0] ** 2 - 2.0])


def Df1(x):
    return np.array([[2.0 * x[0]]])


def cond_fixed(i, x, DFinv_Fx):
    # Original stopping rule: indices 0 through 20 mean 21 updates.
    return i == 20


def cond(i, x, DFinv_Fx):
    return np.max(np.abs(DFinv_Fx)) <= 10 ** (-14)


def f3(p):
    x, y = p
    return np.array([2 * x ** 2 - np.exp(y),
                     -3 * x * y ** 2 - y ** 3 - 3])


def Df3(p):
    x, y = p
    return np.array([[4 * x, -np.exp(y)],
                     [-3 * y ** 2, -6 * x * y - 3 * y ** 2]])


def run_case(label, initial, f, Df, stop):
    history = []
    result = newton(initial, f, Df, stop, history=history)
    residual = np.max(np.abs(f(result)))
    final_step = np.max(np.abs(history[-1][1]))
    print(label)
    print("  initial:", initial)
    print("  solution:", np.array2string(result, precision=15))
    print("  updates:", len(history))
    print(f"  final step infinity norm: {final_step:.6e}")
    print(f"  residual infinity norm: {residual:.6e}")
    return result


def main():
    run_case("Task 1", [2.0], f1, Df1, cond_fixed)
    run_case("Task 2", [2.0], f1, Df1, cond)
    run_case("Task 3", [4.0, 5.0], f3, Df3, cond)
    # Several distinct initial points for task 4.
    for initial in ([1.0, -2.0], [4.0, -5.0], [1.0, 1.0],
                    [-1.0, -1.0], [-4.0, -5.0], [2.0, -2.0]):
        run_case("Task 4", initial, f3, Df3, cond)


if __name__ == "__main__":
    main()
