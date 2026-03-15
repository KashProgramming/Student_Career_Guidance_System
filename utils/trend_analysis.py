import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from numpy import float32
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


HISTORY_FILE = "prediction_history.json"


def save_prediction(
    user_input: Dict,
    placement_prob: float,
    salary_pred: float,
    student_name: Optional[str] = None
) -> bool:
    try:
        # Load existing history
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Create new record
        record = {
            "timestamp": datetime.now().isoformat(),
            "student_name": student_name or "Anonymous",
            "placement_prob": placement_prob,
            "salary_pred": str(salary_pred),
            "data": user_input
        }
        
        history.append(record)
        
        # Save updated history
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        return False


def load_history() -> List[Dict]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        return []


def get_history_dataframe() -> pd.DataFrame:
    history = load_history()
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Flatten student data
    if 'data' in df.columns:
        data_df = pd.json_normalize(df['data'])
        df = pd.concat([df.drop('data', axis=1), data_df], axis=1)
    df['salary_pred']=df['salary_pred'].astype("float32")
    return df


def plot_placement_trend() -> go.Figure:
    df = get_history_dataframe()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.line(
        df,
        x='timestamp',
        y='placement_prob',
        title='Placement Probability Trend',
        labels={'placement_prob': 'Placement Probability', 'timestamp': 'Date'},
        markers=True
    )
    
    fig.update_traces(line_color='#3498db', line_width=3)
    fig.update_layout(
        yaxis_tickformat='.0%',
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    
    # Add average line
    avg_prob = df['placement_prob'].mean()
    fig.add_hline(
        y=avg_prob,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {avg_prob:.1%}"
    )
    
    return fig


def plot_salary_trend() -> go.Figure:
    df = get_history_dataframe()
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.line(
        df,
        x='timestamp',
        y='salary_pred',
        title='Expected Salary Trend',
        labels={'salary_pred': 'Expected Salary (LPA)', 'timestamp': 'Date'},
        markers=True
    )
    
    fig.update_traces(line_color='#27ae60', line_width=3)
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    
    # Add average line
    avg_salary = df['salary_pred'].mean()
    fig.add_hline(
        y=avg_salary,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {avg_salary:.2f} LPA"
    )
    
    return fig


def plot_feature_distribution(feature: str) -> go.Figure:
    df = get_history_dataframe()
    
    if df.empty or feature not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this feature",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = px.histogram(
        df,
        x=feature,
        title=f'Distribution of {feature.replace("_", " ").title()}',
        labels={feature: feature.replace('_', ' ').title()},
        nbins=20
    )
    
    fig.update_traces(marker_color='#9b59b6')
    fig.update_layout(
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    
    return fig


def get_summary_stats() -> Dict:
    df = get_history_dataframe()
    
    if df.empty:
        return {
            "total_predictions": 0,
            "avg_placement_prob": 0,
            "avg_salary": 0,
            "high_risk_count": 0,
            "placed_predictions": 0
        }
    
    df['salary_pred']=df['salary_pred'].astype("float32")
    stats = {
        "total_predictions": len(df),
        "avg_placement_prob": df['placement_prob'].mean(),
        "avg_salary": df['salary_pred'].mean(),
        "high_risk_count": (df['placement_prob'] < 0.4).sum(),
        "placed_predictions": (df['placement_prob'] >= 0.7).sum()
    }
    
    return stats


def compare_current_with_history(current_prob: float, current_salary: float) -> Dict:
    df = get_history_dataframe()
    df['salary_pred']=df['salary_pred'].astype("float32")
    if df.empty:
        return {
            "prob_percentile": None,
            "salary_percentile": None,
            "better_than_avg_prob": None,
            "better_than_avg_salary": None
        }
    
    prob_percentile = (df['placement_prob'] < current_prob).sum() / len(df) * 100
    salary_percentile = (df['salary_pred'] < current_salary).sum() / len(df) * 100
    
    avg_prob = df['placement_prob'].mean()
    avg_salary = df['salary_pred'].mean()
    
    return {
        "prob_percentile": prob_percentile,
        "salary_percentile": salary_percentile,
        "better_than_avg_prob": current_prob > avg_prob,
        "better_than_avg_salary": current_salary > avg_salary,
        "prob_diff_from_avg": current_prob - avg_prob,
        "salary_diff_from_avg": current_salary - avg_salary
    }


def display_trend_analysis():
    st.header("📈 Historical Trend Analysis")
    
    df = get_history_dataframe()
    
    if df.empty:
        st.warning("No historical data available. Make predictions to start tracking trends!")
        return
    
    # Summary statistics
    stats = get_summary_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", stats["total_predictions"])
    
    with col2:
        st.metric("Avg Placement Prob", f"{stats['avg_placement_prob']*100:.1f}%")
    
    with col3:
        st.metric("Avg Salary", f"{stats['avg_salary']:.2f} LPA")
    
    with col4:
        st.metric("High Risk Students", stats["high_risk_count"])
    
    st.markdown("---")
    
    # Trend plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_placement_trend(), width="stretch")
    
    with col2:
        st.plotly_chart(plot_salary_trend(), width="stretch")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    numeric_features = [
        'internships_count', 'projects_count', 'certifications_count',
        'aptitude_score', 'leadership_roles', 'extracurricular_activities'
    ]
    
    available_features = [f for f in numeric_features if f in df.columns]
    
    if available_features:
        selected_feature = st.selectbox("Select feature to analyze", available_features)
        st.plotly_chart(plot_feature_distribution(selected_feature), width="stretch")
    
    # Data table
    with st.expander("📊 View All Historical Data"):
        display_df = df[['timestamp', 'student_name', 'placement_prob', 'salary_pred']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['placement_prob'] = display_df['placement_prob'].apply(lambda x: f"{x*100:.1f}%")
        display_df['salary_pred'] = display_df['salary_pred'].apply(lambda x: f"{x:.2f} LPA")
        
        st.dataframe(display_df, width="stretch")
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.success("History cleared successfully!")
            st.rerun()
