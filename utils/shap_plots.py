"""
Interactive SHAP visualization utilities
"""

import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import BytesIO


def create_waterfall_plot(model, X_input: pd.DataFrame) -> BytesIO:
    """
    Create SHAP waterfall plot showing feature contributions.
    
    Args:
        model: Trained classifier
        X_input: Preprocessed input DataFrame
        
    Returns:
        BytesIO buffer containing the plot image
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)
    
    # For binary classification, use positive class
    if len(shap_values.shape) > 2:
        shap_values = shap_values[:, :, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf


def create_force_plot(model, X_input: pd.DataFrame) -> str:
    """
    Create SHAP force plot as HTML.
    
    Args:
        model: Trained classifier
        X_input: Preprocessed input DataFrame
        
    Returns:
        HTML string for the force plot
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)
    
    # For binary classification
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
        base_value = explainer.expected_value[1]
        shap_vals = shap_values.values[0, :, 1]
    else:
        base_value = explainer.expected_value
        shap_vals = shap_values.values[0]
    
    force_plot = shap.force_plot(
        base_value,
        shap_vals,
        X_input.iloc[0],
        matplotlib=False
    )
    
    return shap.getjs() + force_plot.html()


def create_bar_plot(model, X_input: pd.DataFrame) -> BytesIO:
    """
    Create SHAP bar plot showing feature importance.
    
    Args:
        model: Trained classifier
        X_input: Preprocessed input DataFrame
        
    Returns:
        BytesIO buffer containing the plot image
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)
    
    # For binary classification
    if len(shap_values.shape) > 2:
        shap_values = shap_values[:, :, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values[0], max_display=12, show=False)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return buf


def display_shap_plots(model, X_input: pd.DataFrame):
    """
    Display all SHAP plots in Streamlit.
    
    Args:
        model: Trained classifier
        X_input: Preprocessed input DataFrame
    """
    st.subheader("📊 Interactive SHAP Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Waterfall Plot", "Force Plot", "Bar Chart"])
    
    with tab1:
        st.markdown("**How features push prediction from base value**")
        try:
            waterfall_buf = create_waterfall_plot(model, X_input)
            st.image(waterfall_buf, width="stretch")
            st.caption("Red = increases placement probability | Blue = decreases placement probability")
        except Exception as e:
            st.error(f"Error creating waterfall plot: {str(e)}")
    
    with tab2:
        st.markdown("**Interactive force plot showing feature contributions**")
        try:
            force_html = create_force_plot(model, X_input)
            st.components.v1.html(force_html, height=300, scrolling=True)
            st.caption("Red = positive contribution | Blue = negative contribution")
        except Exception as e:
            st.error(f"Error creating force plot: {str(e)}")
    
    with tab3:
        st.markdown("**Feature importance ranking**")
        try:
            bar_buf = create_bar_plot(model, X_input)
            st.image(bar_buf, width="stretch")
            st.caption("Absolute SHAP values showing feature impact magnitude")
        except Exception as e:
            st.error(f"Error creating bar plot: {str(e)}")
