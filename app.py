import streamlit as st
import pickle
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocessing import preprocess_input, validate_input
from utils.shap_utils import get_feature_impacts, get_top_weaknesses, get_top_strengths
from utils.recommend import generate_recommendations, format_recommendation
from utils.shap_plots import display_shap_plots, create_waterfall_plot
from utils.pdf_generator import create_pdf_report
from utils.trend_analysis import (
    save_prediction, display_trend_analysis, 
    compare_current_with_history, get_history_dataframe
)


@st.cache_resource
def load_models():
    """Load trained models."""
    with open("models/gradient_boost_model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/xg_boost_reg.pkl", "rb") as f:
        reg = pickle.load(f)
    return clf, reg


def get_risk_tier(prob: float) -> tuple[str, str]:
    """Determine risk tier based on placement probability."""
    if prob >= 0.85:
        return "Low Risk", "🟢"
    elif prob >= 0.4:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"


def main():
    st.set_page_config(
        page_title="Career Guidance System",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = None
    if 'placement_prob' not in st.session_state:
        st.session_state.placement_prob = None
    if 'salary_pred' not in st.session_state:
        st.session_state.salary_pred = None
    if 'risk_tier' not in st.session_state:
        st.session_state.risk_tier = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'impact_df' not in st.session_state:
        st.session_state.impact_df = None
    if 'X_input' not in st.session_state:
        st.session_state.X_input = None
    if 'student_name' not in st.session_state:
        st.session_state.student_name = None
    
    st.title("🎓 Career Guidance & Placement Prediction System")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Make Prediction", "📈 View Trends"],
        index=0
    )
    
    # Show trends page
    if page == "📈 View Trends":
        display_trend_analysis()
        return
    
    st.sidebar.markdown("---")
    
    # Load models
    clf_model, reg_model = load_models()
    
    # Sidebar for input
    st.sidebar.header("Student Information")
    
    # Optional student name for tracking
    student_name = st.sidebar.text_input("Student Name (optional)", placeholder="For tracking only", value=st.session_state.get('student_name', ''))
    st.session_state.student_name = student_name if student_name else None
    st.sidebar.markdown("---")
    
    # Collect raw inputs
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
        ["Engineering", "Business", "Science", "Arts", "Other"]
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
    
    # Predict button
    if st.sidebar.button("🔮 Get Predictions", type="primary"):
        # Validate input
        is_valid, error_msg = validate_input(user_input)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        # Preprocess
        X_input = preprocess_input(user_input)
        
        # Predictions
        placement_prob = clf_model.predict_proba(X_input)[0][1]
        salary_pred = reg_model.predict(X_input)[0]
        
        risk_tier, risk_emoji = get_risk_tier(placement_prob)
        
        # Generate recommendations and analysis
        recommendations = generate_recommendations(
            X_input,
            clf_model,
            reg_model,
            placement_prob,
            salary_pred,
            top_n=5
        )
        
        impact_df = get_feature_impacts(clf_model, X_input)
        
        # Store in session state
        st.session_state.prediction_made = True
        st.session_state.user_input = user_input
        st.session_state.placement_prob = placement_prob
        st.session_state.salary_pred = salary_pred
        st.session_state.risk_tier = risk_tier
        st.session_state.risk_emoji = risk_emoji
        st.session_state.recommendations = recommendations
        st.session_state.impact_df = impact_df
        st.session_state.X_input = X_input
    
    # Display results if prediction was made
    if st.session_state.prediction_made:
        # Retrieve from session state
        placement_prob = st.session_state.placement_prob
        salary_pred = st.session_state.salary_pred
        risk_tier = st.session_state.risk_tier
        risk_emoji = st.session_state.risk_emoji
        recommendations = st.session_state.recommendations
        impact_df = st.session_state.impact_df
        X_input = st.session_state.X_input
        user_input = st.session_state.user_input
        
        # Display results
        st.success("✅ Predictions Generated Successfully!")
        
        # Section 1: Summary
        st.header("📊 Prediction Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Placement Probability",
                f"{placement_prob*100:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Expected Salary",
                f"{salary_pred:.2f} LPA",
                delta=None
            )
        
        with col3:
            st.metric(
                "Risk Tier",
                risk_tier,
                delta=None
            )
            st.markdown(f"### {risk_emoji}")
        
        st.markdown("---")
        
        # Section 2: Recommendations
        st.header("🎯 Top Improvement Areas")
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} - {rec['feature'].replace('_', ' ').title()}", expanded=(i <= 3)):
                st.markdown(format_recommendation(rec))
        
        st.markdown("---")
        
        # Historical comparison
        if len(get_history_dataframe()) > 0:
            st.header("📊 Historical Comparison")
            
            comparison = compare_current_with_history(placement_prob, salary_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Placement Probability Percentile",
                    f"{comparison['prob_percentile']:.0f}%",
                    delta=f"{comparison['prob_diff_from_avg']*100:+.1f}% vs avg" if comparison['better_than_avg_prob'] is not None else None
                )
                
            with col2:
                st.metric(
                    "Salary Percentile",
                    f"{comparison['salary_percentile']:.0f}%",
                    delta=f"{comparison['salary_diff_from_avg']:+.2f} LPA vs avg" if comparison['better_than_avg_salary'] is not None else None
                )
            
            if comparison['prob_percentile']:
                if comparison['prob_percentile'] >= 75:
                    st.success("🌟 This student is performing better than 75% of historical predictions!")
                elif comparison['prob_percentile'] >= 50:
                    st.info("📈 This student is performing above average compared to historical data.")
                else:
                    st.warning("⚠️ This student may need additional support compared to historical trends.")
        
        st.markdown("---")
        
        # Section 3: SHAP Analysis
        st.header("🔍 Feature Contribution Analysis")
        
        # Interactive SHAP plots
        display_shap_plots(clf_model, X_input)
        
        st.markdown("---")
        
        # Traditional SHAP text summary
        st.subheader("Feature Impact Summary")
        
        impact_df = get_feature_impacts(clf_model, X_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💪 Top Strengths")
            strengths = get_top_strengths(impact_df, top_n=5)
            
            for _, row in strengths.iterrows():
                feature_display = row['feature'].replace('_', ' ').title()
                st.markdown(f"- **{feature_display}**: +{row['impact']:.3f}")
        
        with col2:
            st.subheader("⚠️ Top Weaknesses")
            weaknesses = get_top_weaknesses(impact_df, top_n=5)
            
            for _, row in weaknesses.iterrows():
                feature_display = row['feature'].replace('_', ' ').title()
                st.markdown(f"- **{feature_display}**: {row['impact']:.3f}")
        
        st.info(
            "**Interpretation**: Positive values indicate features that increase your placement probability "
            "relative to the average. Negative values indicate areas where you're below average."
        )
        
        st.markdown("---")
        
        # Export and Save Options
        st.header("💾 Export & Save")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PDF Export
            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True, key="generate_pdf"):
                with st.spinner("Generating PDF report..."):
                    try:
                        # Create waterfall plot for PDF
                        waterfall_buf = create_waterfall_plot(clf_model, X_input)
                        
                        # Generate PDF
                        pdf_buffer = create_pdf_report(
                            user_input,
                            placement_prob,
                            salary_pred,
                            risk_tier,
                            recommendations,
                            get_top_strengths(impact_df),
                            get_top_weaknesses(impact_df),
                            waterfall_buf
                        )
                        
                        st.session_state.pdf_buffer = pdf_buffer
                        st.success("✅ PDF generated successfully!")
                    
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
            
            # Download button (separate, always visible if PDF exists)
            if 'pdf_buffer' in st.session_state and st.session_state.pdf_buffer:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_buffer,
                    file_name=f"career_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf"
                )
        
        with col2:
            # Save to history
            if st.button("💾 Save to History", type="secondary", use_container_width=True, key="save_history"):
                if save_prediction(user_input, placement_prob, salary_pred, st.session_state.student_name):
                    st.success("✅ Prediction saved to history!")
                    st.session_state.history_saved = True
                else:
                    st.error("❌ Failed to save prediction")
        
        st.markdown("---")
        
        # Disclaimer
        st.caption(
            "⚠️ **Disclaimer**: Predictions are probabilistic estimates based on historical data. "
            "They are not guarantees of actual placement or salary outcomes."
        )
    
    else:
        st.info("👈 Fill in your details in the sidebar and click 'Get Predictions' to start!")


if __name__ == "__main__":
    main()