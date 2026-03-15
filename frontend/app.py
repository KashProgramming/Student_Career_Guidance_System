from __future__ import annotations

import base64
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.api_client import post
from utils.preprocessing import validate_input
from utils.recommend import format_recommendation
from utils.pdf_generator import create_pdf_report
from utils.trend_analysis import (
    save_prediction,
    display_trend_analysis,
    compare_current_with_history,
    get_history_dataframe,
)


def main() -> None:
    st.set_page_config(
        page_title="Career Guidance System",
        page_icon="CG",
        layout="wide",
    )

    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    if "user_input" not in st.session_state:
        st.session_state.user_input = None
    if "placement_prob" not in st.session_state:
        st.session_state.placement_prob = None
    if "salary_pred" not in st.session_state:
        st.session_state.salary_pred = None
    if "risk_tier" not in st.session_state:
        st.session_state.risk_tier = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    if "impact_df" not in st.session_state:
        st.session_state.impact_df = None
    if "strengths_df" not in st.session_state:
        st.session_state.strengths_df = None
    if "weaknesses_df" not in st.session_state:
        st.session_state.weaknesses_df = None
    if "waterfall_base64" not in st.session_state:
        st.session_state.waterfall_base64 = None
    if "bar_base64" not in st.session_state:
        st.session_state.bar_base64 = None
    if "student_name" not in st.session_state:
        st.session_state.student_name = None

    st.title("Career Guidance & Placement Prediction System")
    st.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Make Prediction", "View Trends"],
        index=0,
    )

    if page == "View Trends":
        display_trend_analysis()
        return

    st.sidebar.markdown("---")
    st.sidebar.header("Student Information")

    student_name = st.sidebar.text_input(
        "Student Name (optional)",
        placeholder="For tracking only",
        value=st.session_state.get("student_name", ""),
    )
    st.session_state.student_name = student_name if student_name else None

    user_input = {}

    st.sidebar.subheader("Demographics")
    user_input["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    user_input["city_tier"] = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    st.sidebar.subheader("Academic Background")
    user_input["ssc_board"] = st.sidebar.selectbox("SSC Board", ["CBSE", "ICSE", "State"])
    user_input["hsc_board"] = st.sidebar.selectbox("HSC Board", ["CBSE", "ICSE", "State"])
    user_input["hsc_stream"] = st.sidebar.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
    user_input["hsc_percentage"] = st.sidebar.slider("HSC Percentage", 0.0, 100.0, 75.0, 0.1)
    user_input["degree_field"] = st.sidebar.selectbox(
        "Degree Field",
        ["Engineering", "Business", "Science", "Arts", "Other"],
    )
    user_input["degree_percentage"] = st.sidebar.slider("Degree Percentage", 0.0, 100.0, 70.0, 0.1)
    user_input["backlogs"] = st.sidebar.number_input("Backlogs", 0, 10, 0)

    st.sidebar.subheader("Skills & Scores")
    user_input["technical_skills_score"] = st.sidebar.slider("Technical Skills Score", 0.0, 10.0, 6.0, 0.1)
    user_input["soft_skills_score"] = st.sidebar.slider("Soft Skills Score", 0.0, 10.0, 6.0, 0.1)
    user_input["aptitude_score"] = st.sidebar.slider("Aptitude Score", 0.0, 100.0, 60.0, 0.1)

    st.sidebar.subheader("Experience & Activities")
    user_input["internships_count"] = st.sidebar.number_input("Internships", 0, 10, 1)
    user_input["projects_count"] = st.sidebar.number_input("Projects", 0, 20, 2)
    user_input["certifications_count"] = st.sidebar.number_input("Certifications", 0, 10, 1)
    user_input["work_experience_months"] = st.sidebar.number_input("Work Experience (months)", 0, 60, 0)
    user_input["leadership_roles"] = st.sidebar.number_input("Leadership Roles", 0, 10, 0)
    user_input["extracurricular_activities"] = st.sidebar.number_input("Extracurricular Activities", 0, 20, 2)

    if st.sidebar.button("Get Predictions", type="primary"):
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return

        payload = {**user_input, "student_name": st.session_state.student_name}

        try:
            with st.spinner("Calling prediction service..."):
                prediction = post("/predict", payload)
                explain = post("/explain", payload)
                recommendations = post("/recommendations", payload)

            st.session_state.prediction_made = True
            st.session_state.user_input = user_input
            st.session_state.placement_prob = prediction["placement_probability"]
            st.session_state.salary_pred = prediction["expected_salary"]
            st.session_state.risk_tier = prediction["risk_tier"]
            st.session_state.recommendations = recommendations["recommendations"]

            st.session_state.impact_df = pd.DataFrame(explain["feature_impacts"])
            st.session_state.strengths_df = pd.DataFrame(explain["strengths"])
            st.session_state.weaknesses_df = pd.DataFrame(explain["weaknesses"])
            st.session_state.waterfall_base64 = explain.get("waterfall_plot_base64")
            st.session_state.bar_base64 = explain.get("bar_plot_base64")

        except Exception as exc:
            st.error(f"API error: {exc}")
            return

    if st.session_state.prediction_made:
        placement_prob = st.session_state.placement_prob
        salary_pred = st.session_state.salary_pred
        risk_tier = st.session_state.risk_tier
        recommendations = st.session_state.recommendations
        impact_df = st.session_state.impact_df
        strengths_df = st.session_state.strengths_df
        weaknesses_df = st.session_state.weaknesses_df
        user_input = st.session_state.user_input

        st.success("Predictions Generated Successfully!")

        st.header("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Placement Probability", f"{placement_prob * 100:.1f}%")

        with col2:
            st.metric("Expected Salary", f"{salary_pred:.2f} LPA")

        with col3:
            st.metric("Risk Tier", risk_tier)

        st.markdown("---")

        st.header("Top Improvement Areas")
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} - {rec['feature'].replace('_', ' ').title()}", expanded=(i <= 3)):
                st.markdown(format_recommendation(rec))

        st.markdown("---")

        if len(get_history_dataframe()) > 0:
            st.header("Historical Comparison")
            comparison = compare_current_with_history(placement_prob, salary_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Placement Probability Percentile",
                    f"{comparison['prob_percentile']:.0f}%",
                    delta=(
                        f"{comparison['prob_diff_from_avg'] * 100:+.1f}% vs avg"
                        if comparison["better_than_avg_prob"] is not None
                        else None
                    ),
                )

            with col2:
                st.metric(
                    "Salary Percentile",
                    f"{comparison['salary_percentile']:.0f}%",
                    delta=(
                        f"{comparison['salary_diff_from_avg']:+.2f} LPA vs avg"
                        if comparison["better_than_avg_salary"] is not None
                        else None
                    ),
                )

            if comparison["prob_percentile"]:
                if comparison["prob_percentile"] >= 75:
                    st.success("This student is performing better than 75% of historical predictions!")
                elif comparison["prob_percentile"] >= 50:
                    st.info("This student is performing above average compared to historical data.")
                else:
                    st.warning("This student may need additional support compared to historical trends.")

        st.markdown("---")

        st.header("Feature Contribution Analysis")

        if st.session_state.waterfall_base64:
            st.subheader("Waterfall Plot")
            st.image(base64.b64decode(st.session_state.waterfall_base64), width="stretch")

        if st.session_state.bar_base64:
            st.subheader("Feature Importance")
            st.image(base64.b64decode(st.session_state.bar_base64), width="stretch")

        st.subheader("Feature Impact Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Strengths")
            if strengths_df is not None and not strengths_df.empty:
                for _, row in strengths_df.iterrows():
                    feature_display = row["feature"].replace("_", " ").title()
                    st.markdown(f"- **{feature_display}**: +{row['impact']:.3f}")

        with col2:
            st.subheader("Top Weaknesses")
            if weaknesses_df is not None and not weaknesses_df.empty:
                for _, row in weaknesses_df.iterrows():
                    feature_display = row["feature"].replace("_", " ").title()
                    st.markdown(f"- **{feature_display}**: {row['impact']:.3f}")

        if impact_df is not None and not impact_df.empty:
            with st.expander("View full feature impacts"):
                st.dataframe(impact_df, width="stretch")

        st.info(
            "**Interpretation**: Positive values indicate features that increase your placement probability "
            "relative to the average. Negative values indicate areas where you're below average."
        )

        st.markdown("---")

        st.header("Export & Save")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate PDF Report", type="primary", width="stretch", key="generate_pdf"):
                with st.spinner("Generating PDF report..."):
                    try:
                        waterfall_buf = None
                        if st.session_state.waterfall_base64:
                            waterfall_buf = BytesIO(base64.b64decode(st.session_state.waterfall_base64))

                        pdf_buffer = create_pdf_report(
                            user_input,
                            placement_prob,
                            salary_pred,
                            risk_tier,
                            recommendations,
                            strengths_df if strengths_df is not None else pd.DataFrame(),
                            weaknesses_df if weaknesses_df is not None else pd.DataFrame(),
                            waterfall_buf,
                        )

                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("PDF generated successfully!")

                    except Exception as exc:
                        st.error(f"Error generating PDF: {exc}")

            if "pdf_buffer" in st.session_state and st.session_state.pdf_buffer:
                st.download_button(
                    label="Download PDF",
                    data=st.session_state.pdf_buffer,
                    file_name=f"career_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="download_pdf",
                )

        with col2:
            if st.button("Save to History", type="secondary", width="stretch", key="save_history"):
                if save_prediction(user_input, placement_prob, salary_pred, st.session_state.student_name):
                    st.success("Prediction saved to history!")
                    st.session_state.history_saved = True
                else:
                    st.error("Failed to save prediction")

        st.markdown("---")

        st.caption(
            "**Disclaimer**: Predictions are probabilistic estimates based on historical data. "
            "They are not guarantees of actual placement or salary outcomes."
        )

    else:
        st.info("Fill in your details in the sidebar and click 'Get Predictions' to start!")


if __name__ == "__main__":
    main()
