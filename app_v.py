# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:46:15 2026

@author: hideki.oka
"""

import streamlit as st
import pandas as pd
import viewer_core as vc


st.set_page_config(
    page_title="Regression Viewer",
    layout="wide"
)

st.title(
    "Regression Viewer"
)

tab_model, tab_vars, tab_profiler, tab_contour = st.tabs(
    [
        "Model",
        "Variables",
        "1D Profiler",
        "2D Contour"
    ]
)

# -------------------
# Model
# -------------------

with tab_model:

    default_model = pd.DataFrame(
        {
            "Term": [
                "Intercept",
                "MSG",
                "Moisture",
                "MSG*Moisture"
            ],
            "Coef": [
                1.0,
                0.2,
                0.3,
                0.05
            ]
        }
    )

    model_df = st.data_editor(
        default_model,
        num_rows="dynamic",
        use_container_width=True
    )

    inverse_method = st.selectbox(
        "Response Transform",
        [
            "none",
            "log",
            "logit"
        ]
    )

    st.session_state.model_df = model_df
    st.session_state.inverse_method = inverse_method

# -------------------
# Variables
# -------------------

with tab_vars:

    if "model_df" in st.session_state:

        coeffs = dict(
            zip(
                st.session_state.model_df["Term"],
                st.session_state.model_df["Coef"]
            )
        )

        if "variable_df" not in st.session_state:

            variables = vc.extract_variables(
                coeffs
            )

            st.session_state.variable_df = pd.DataFrame(
                {
                    "Variable": variables,
                    "Min": [0.0] * len(variables),
                    "Max": [1.0] * len(variables),
                    "Default": [0.5] * len(variables)
                }
            )

        if st.button("Refresh Variables"):

            variables = vc.extract_variables(
                coeffs
            )

            st.session_state.variable_df = pd.DataFrame(
                {
                    "Variable": variables,
                    "Min": [0.0] * len(variables),
                    "Max": [1.0] * len(variables),
                    "Default": [0.5] * len(variables)
                }
            )

            st.rerun()

        variable_df = st.data_editor(
            st.session_state.variable_df,
            use_container_width=True,
            key="variable_editor"
        )

        st.session_state.variable_df = variable_df
        
# -------------------
# 1D Profiler
# -------------------

with tab_profiler:

    if (
        "model_df" in st.session_state
        and
        "variable_df" in st.session_state
    ):

        coeffs = dict(
            zip(
                st.session_state.model_df["Term"],
                st.session_state.model_df["Coef"]
            )
        )

        variable_df = (
            st.session_state.variable_df
        )

        slider_values = {}
        variable_ranges = {}

        for _, row in variable_df.iterrows():

            var = row["Variable"]

            slider_values[var] = st.slider(
                label=var,
                min_value=float(row["Min"]),
                max_value=float(row["Max"]),
                value=float(row["Default"])
            )

            variable_ranges[var] = (
                float(row["Min"]),
                float(row["Max"])
            )

        fig = vc.profiler(
            coeffs,
            slider_values,
            variable_ranges,
            inverse=st.session_state.inverse_method
        )

        st.pyplot(fig)

# -------------------
# 2D Contour
# -------------------

with tab_contour:

    if (
        "model_df" in st.session_state
        and
        "variable_df" in st.session_state
    ):

        coeffs = dict(
            zip(
                st.session_state.model_df["Term"],
                st.session_state.model_df["Coef"]
            )
        )

        variable_df = (
            st.session_state.variable_df
        )

        slider_values = {}
        variable_ranges = {}

        for _, row in variable_df.iterrows():

            var = row["Variable"]

            slider_values[var] = st.number_input(
                var,
                value=float(row["Default"])
            )

            variable_ranges[var] = (
                float(row["Min"]),
                float(row["Max"])
            )

        if st.button(
            "Update Contours"
        ):

            fig = vc.profiler_2d_all(
                coeffs,
                slider_values,
                variable_ranges,
                inverse=st.session_state.inverse_method
            )

            st.pyplot(fig)