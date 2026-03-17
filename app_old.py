import streamlit as st
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocessing import preprocess_input, validate_input
from utils.shap_utils import get_feature_impacts, get_top_weaknesses, get_top_strengths
from utils.recommend import generate_recommendations, format_recommendation


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
    if prob >= 0.8:
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
    
    st.title("🎓 Career Guidance & Placement Prediction System")
    st.markdown("---")
    
    # Load models
    clf_model, reg_model = load_models()
    
    # Sidebar for input
    st.sidebar.header("Student Information")
    
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
        
        recommendations = generate_recommendations(
            X_input,
            clf_model,
            reg_model,
            placement_prob,
            salary_pred,
            top_n=5
        )
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} - {rec['feature'].replace('_', ' ').title()}", expanded=(i <= 3)):
                st.markdown(format_recommendation(rec))
        
        st.markdown("---")
        
        # Section 3: SHAP Analysis
        st.header("🔍 Feature Contribution Analysis")
        
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
        
        # Disclaimer
        st.caption(
            "⚠️ **Disclaimer**: Predictions are probabilistic estimates based on historical data. "
            "They are not guarantees of actual placement or salary outcomes."
        )
    
    else:
        st.info("👈 Fill in your details in the sidebar and click 'Get Predictions' to start!")


if __name__ == "__main__":
    main()
