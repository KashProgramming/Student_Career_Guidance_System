
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import pandas as pd
from io import BytesIO
from typing import List, Dict


def create_pdf_report(
    student_data: Dict,
    placement_prob: float,
    salary_pred: float,
    risk_tier: str,
    recommendations: List[Dict],
    strengths: pd.DataFrame,
    weaknesses: pd.DataFrame,
    waterfall_buf: BytesIO = None
) -> BytesIO:
    """
    Generate comprehensive PDF report.
    
    Args:
        student_data: Original student input data
        placement_prob: Predicted placement probability
        salary_pred: Predicted salary
        risk_tier: Risk classification
        recommendations: List of recommendations
        strengths: Top strengths DataFrame
        weaknesses: Top weaknesses DataFrame
        waterfall_buf: Optional SHAP waterfall plot image
        
    Returns:
        BytesIO buffer containing PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Career Guidance Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Prediction Summary
    story.append(Paragraph("Prediction Summary", heading_style))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Placement Probability', f"{placement_prob*100:.1f}%"],
        ['Expected Salary', f"{salary_pred:.2f} LPA"],
        ['Risk Tier', risk_tier]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Top Recommendations
    story.append(Paragraph("Top 5 Improvement Recommendations", heading_style))
    
    for i, rec in enumerate(recommendations[:5], 1):
        feature_display = rec['feature'].replace('_', ' ').title()
        rec_text = f"<b>{i}. Increase {feature_display} by {rec['increment']:.1f}</b><br/>"
        rec_text += f"→ {rec['delta_prob']:+.1f}% placement probability<br/>"
        rec_text += f"→ {rec['delta_salary']:+.2f} LPA expected salary"
        
        story.append(Paragraph(rec_text, styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Strengths and Weaknesses
    story.append(Paragraph("Feature Analysis", heading_style))
    
    col1_data = [['Top Strengths', 'Impact']]
    for _, row in strengths.head(5).iterrows():
        col1_data.append([
            row['feature'].replace('_', ' ').title(),
            f"+{row['impact']:.3f}"
        ])
    
    col2_data = [['Top Weaknesses', 'Impact']]
    for _, row in weaknesses.head(5).iterrows():
        col2_data.append([
            row['feature'].replace('_', ' ').title(),
            f"{row['impact']:.3f}"
        ])
    
    # Combine into side-by-side tables
    analysis_data = []
    for i in range(max(len(col1_data), len(col2_data))):
        row = []
        if i < len(col1_data):
            row.extend(col1_data[i])
        else:
            row.extend(['', ''])
        row.append('')  # Spacer column
        if i < len(col2_data):
            row.extend(col2_data[i])
        else:
            row.extend(['', ''])
        analysis_data.append(row)
    
    analysis_table = Table(analysis_data, colWidths=[2*inch, 0.8*inch, 0.2*inch, 2*inch, 0.8*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#27ae60')),
        ('BACKGROUND', (3, 0), (4, 0), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (3, 0), (4, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (1, -1), 0.5, colors.grey),
        ('GRID', (3, 0), (4, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (1, -1), [colors.white, colors.lightgreen]),
        ('ROWBACKGROUNDS', (3, 1), (4, -1), [colors.white, colors.lightpink])
    ]))
    
    story.append(analysis_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add SHAP plot if provided
    if waterfall_buf:
        story.append(PageBreak())
        story.append(Paragraph("SHAP Waterfall Analysis", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        waterfall_buf.seek(0)
        img = Image(waterfall_buf, width=6.5*inch, height=4*inch)
        story.append(img)
    
    # Student Profile (on new page)
    story.append(PageBreak())
    story.append(Paragraph("Student Profile", heading_style))
    
    profile_data = [['Attribute', 'Value']]
    
    # Demographics
    profile_data.append(['Gender', student_data.get('gender', 'N/A')])
    profile_data.append(['City Tier', student_data.get('city_tier', 'N/A')])
    
    # Academic
    profile_data.append(['HSC Board', student_data.get('hsc_board', 'N/A')])
    profile_data.append(['HSC Stream', student_data.get('hsc_stream', 'N/A')])
    profile_data.append(['HSC %', f"{student_data.get('hsc_percentage', 0):.1f}%"])
    profile_data.append(['Degree Field', student_data.get('degree_field', 'N/A')])
    profile_data.append(['Degree %', f"{student_data.get('degree_percentage', 0):.1f}%"])
    profile_data.append(['Backlogs', str(student_data.get('backlogs', 0))])
    
    # Skills
    profile_data.append(['Technical Skills', f"{student_data.get('technical_skills_score', 0):.1f}/10"])
    profile_data.append(['Soft Skills', f"{student_data.get('soft_skills_score', 0):.1f}/10"])
    profile_data.append(['Aptitude Score', f"{student_data.get('aptitude_score', 0):.1f}/100"])
    
    # Experience
    profile_data.append(['Internships', str(student_data.get('internships_count', 0))])
    profile_data.append(['Projects', str(student_data.get('projects_count', 0))])
    profile_data.append(['Certifications', str(student_data.get('certifications_count', 0))])
    profile_data.append(['Work Experience', f"{student_data.get('work_experience_months', 0)} months"])
    profile_data.append(['Leadership Roles', str(student_data.get('leadership_roles', 0))])
    profile_data.append(['Extracurriculars', str(student_data.get('extracurricular_activities', 0))])
    
    profile_table = Table(profile_data, colWidths=[3*inch, 3*inch])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(profile_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Spacer(1, 0.5*inch))
    disclaimer_text = "<i>Disclaimer: Predictions are probabilistic estimates based on historical data. " \
                     "They are not guarantees of actual placement or salary outcomes. " \
                     "Actual results depend on multiple factors including market conditions, " \
                     "interview performance, and company requirements.</i>"
    story.append(Paragraph(disclaimer_text, styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer
