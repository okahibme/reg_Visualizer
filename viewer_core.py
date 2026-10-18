# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:45:11 2026

@author: hideki.oka
"""

import numpy as np
import matplotlib.pyplot as plt
import re

from itertools import combinations


def extract_variables(coeffs):

    variables = set()

    for term in coeffs:

        if term.lower() == "intercept":
            continue

        for part in term.split("*"):

            part = part.strip()

            m = re.fullmatch(
                r"(log|log10|ln)\((.+)\)",
                part,
                flags=re.IGNORECASE
            )

            if m:

                variables.add(
                    m.group(2).strip()
                )

                continue

            if "^" in part:

                variables.add(
                    part.split("^")[0].strip()
                )

                continue

            variables.add(part)

    return sorted(variables)


def evaluate_term(
    term,
    variables
):

    term = term.strip()

    if term.lower() == "intercept":
        return 1.0

    if "*" in term:

        value = 1.0

        for part in term.split("*"):

            value *= evaluate_term(
                part.strip(),
                variables
            )

        return value

    m = re.fullmatch(
        r"log\((.+)\)",
        term,
        flags=re.IGNORECASE
    )

    if m:

        var = m.group(1).strip()

        return np.log10(
            variables[var]
        )

    m = re.fullmatch(
        r"log10\((.+)\)",
        term,
        flags=re.IGNORECASE
    )

    if m:

        var = m.group(1).strip()

        return np.log10(
            variables[var]
        )

    m = re.fullmatch(
        r"ln\((.+)\)",
        term,
        flags=re.IGNORECASE
    )

    if m:

        var = m.group(1).strip()

        return np.log(
            variables[var]
        )

    if "^" in term:

        base, power = term.split(
            "^",
            1
        )

        return (
            variables[
                base.strip()
            ]
            **
            float(power)
        )

    return variables[term]


def predict(
    coeffs,
    variables
):

    y = 0.0

    for term, coef in coeffs.items():

        y += (
            coef
            *
            evaluate_term(
                term,
                variables
            )
        )

    return y


def inverse_transform(
    y,
    method="none"
):

    method = method.lower()

    if method == "none":
        return y

    if method == "log":
        return np.exp(y)

    if method == "logit":
        return 1.0 / (
            1.0 + np.exp(-y)
        )

    raise ValueError(
        method
    )


def profiler_1d(
    ax,
    coeffs,
    x_var,
    x_min,
    x_max,
    slider_values,
    inverse="none",
    n_points=100
):

    x = np.linspace(
        x_min,
        x_max,
        n_points
    )

    y = np.zeros(
        n_points
    )

    current_value = slider_values[x_var]

    for i, xi in enumerate(x):

        variables = dict(
            slider_values
        )

        variables[x_var] = xi

        z = predict(
            coeffs,
            variables
        )

        y[i] = inverse_transform(
            z,
            inverse
        )

    current_prediction = inverse_transform(
        predict(
            coeffs,
            slider_values
        ),
        inverse
    )

    ax.plot(x, y)

    ax.axvline(
        current_value,
        color="red",
        ls="--"
    )

    ax.plot(
        current_value,
        current_prediction,
        "ro"
    )

    ax.set_title(x_var)

    ax.grid(True)


def profiler(
    coeffs,
    slider_values,
    variable_ranges,
    inverse="none"
):

    variables = extract_variables(
        coeffs
    )

    n = len(variables)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4 * n, 4),
        sharey=True
    )

    if n == 1:
        axes = [axes]

    for ax, var in zip(
        axes,
        variables
    ):

        xmin, xmax = variable_ranges[var]

        profiler_1d(
            ax=ax,
            coeffs=coeffs,
            x_var=var,
            x_min=xmin,
            x_max=xmax,
            slider_values=slider_values,
            inverse=inverse
        )

    plt.tight_layout()

    return fig


def profiler_2d(
    ax,
    coeffs,
    x_var,
    y_var,
    x_range,
    y_range,
    slider_values,
    inverse="none",
    n_points=100
):

    x = np.linspace(
        x_range[0],
        x_range[1],
        n_points
    )

    y = np.linspace(
        y_range[0],
        y_range[1],
        n_points
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    variables = dict(
        slider_values
    )

    variables[x_var] = X
    variables[y_var] = Y

    Z = predict(
        coeffs,
        variables
    )

    Z = inverse_transform(
        Z,
        inverse
    )

    cs = ax.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap="viridis"
    )

    ''' #点は現時点で不要
    ax.plot(
        slider_values[x_var],
        slider_values[y_var],
        "ro"
    )
    '''
    ax.set_xlabel(x_var)

    ax.set_ylabel(y_var)

    ax.set_title(
        f"{x_var} vs {y_var}"
    )

    return cs


def profiler_2d_all(
    coeffs,
    slider_values,
    variable_ranges,
    inverse="none",
    n_points=100
):

    variables = extract_variables(
        coeffs
    )

    pairs = list(
        combinations(
            variables,
            2
        )
    )

    n = len(pairs)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 4),
        constrained_layout=True
    )

    if n == 1:
        axes = [axes]

    last_cs = None

    for ax, (
        x_var,
        y_var
    ) in zip(
        axes,
        pairs
    ):

        last_cs = profiler_2d(
            ax=ax,
            coeffs=coeffs,
            x_var=x_var,
            y_var=y_var,
            x_range=variable_ranges[x_var],
            y_range=variable_ranges[y_var],
            slider_values=slider_values,
            inverse=inverse,
            n_points=n_points
        )

    cbar = fig.colorbar(
        last_cs,
        ax=axes,
        location="right",
        shrink=0.9
    )

    cbar.set_label(
        "Prediction"
    )

    return fig
