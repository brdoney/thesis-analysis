from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import curve_fit


def r2(y_exp: npt.NDArray[Any], y_pred: npt.NDArray[Any]) -> float:
    return 1.0 - (np.var(y_pred - y_exp) / np.var(y_exp))


def fit_exponential(series: "pd.Series[int]") -> None:
    def exponential(x, a, b):
        return a * np.exp(b * x)

    x_data = np.arange(len(series)) + 1
    y_data = series.values
    params, _ = curve_fit(
        exponential,
        x_data,
        y_data,
        p0=(1, -0.1),
        bounds=([-np.inf, -10], [np.inf, -0.0001]),
    )  # p0 is the initial guess for [a, b]

    res = exponential(x_data, *params)
    plt.plot(
        x_data - 1,
        res,
        color="yellow",
        label=f"$y = {params[0]:.2f} e^{{{params[1]:.2f}x}}$",
    )
    plt.legend()

    print("Exponential", r2(y_data, res))


def fit_zipf(series: "pd.Series[int]") -> None:
    num_recs = series.sum()

    def zipf_function(x, a):
        return stats.zipf.pmf(x, a) * num_recs  # Scale the PDF to match the total count

    # Fit Zipf distribution to the data
    x_data = np.arange(len(series)) + 1
    y_data = series.values
    params, _ = curve_fit(
        zipf_function, x_data, y_data, p0=[1.5]
    )  # Initial guess for parameter `a`

    res = zipf_function(x_data, params[0])
    plt.plot(x_data - 1, res, color="red", label=f"Zipf: $a = {params[0]:.2f}$")
    plt.legend()

    print("Zipf", r2(y_data, res))


def fit_zipfian(series: "pd.Series[int]") -> None:
    num_recs = series.sum()

    def zipf_function(x, a):
        return (
            stats.zipfian.pmf(x, a, len(series)) * num_recs
        )  # Scale the PDF to match the total count

    # Fit Zipf distribution to the data
    x_data = np.arange(len(series)) + 1
    y_data = series.values
    params, _ = curve_fit(
        zipf_function, x_data, y_data, p0=[1.5]
    )  # Initial guess for parameter `a`

    res = zipf_function(x_data, params[0])
    plt.plot(x_data - 1, res, color="purple", label=f"Zipfian: $a = {params[0]:.2f}$")
    plt.legend()

    print("Zipfian", r2(y_data, res))


def fit_log(series: "pd.Series[int]") -> None:
    def logarithmic(x, a, b, c):
        return a + b * np.log(x) + c * np.log(x) ** 2

    x_data = np.arange(len(series)) + 1
    y_data = series.values
    params, _ = curve_fit(
        logarithmic, x_data, y_data, p0=[1, 1, 1]
    )  # Initial guess for parameter `a`

    res = logarithmic(x_data, *params)
    plt.plot(
        x_data - 1,
        res,
        color="red",
        label=f"$y = {params[0]:.2f}+ {params[1]:.2f}\\log{{x}}+{params[2]:.2f}\\log^2{{x}}$",
    )
    plt.legend()

    print("Log", r2(y_data, res))
