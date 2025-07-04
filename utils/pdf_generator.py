"""
PDF generator utility for plant disease detection.
Creates detailed PDF reports with analysis results and recommendations.
"""

from fpdf import FPDF
import os
from datetime import datetime
from PIL import Image
import io
import streamlit as st

class PlantDiseaseReport(FPDF):
    """Custom PDF class for plant disease reports"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Plant Disease Detection Report', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        
    def chapter_title(self, title):
        """Add chapter title"""
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, body):
        """Add chapter body text"""
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def generate_pdf_report(prediction_result, disease_info, uploaded_image, sections=None):
    """
    Generate a comprehensive PDF report for plant disease analysis.
    
    Args:
        prediction_result: Dictionary containing prediction results
        disease_info: Dictionary containing disease information
        uploaded_image: Uploaded image file
        sections: List of sections to include in report
        
    Returns:
        str: Path to generated PDF file
    """
    try:
        # Create PDF object
        pdf = PlantDiseaseReport()
        pdf.alias_nb_pages()
        
        # Set default sections if none provided
        if sections is None:
            sections = ["Disease Analysis", "Treatment Recommendations", "Prevention Tips"]
        
        # Add title page
        add_title_page(pdf, prediction_result)
        
        # Add analysis summary
        if "Disease Analysis" in sections:
            add_analysis_summary(pdf, prediction_result, disease_info)
        
        # Add treatment recommendations
        if "Treatment Recommendations" in sections:
            add_treatment_recommendations(pdf, disease_info)
        
        # Add prevention tips
        if "Prevention Tips" in sections:
            add_prevention_tips(pdf, disease_info)
        
        # Add image analysis
        if "Image Analysis" in sections and uploaded_image:
            add_image_analysis(pdf, uploaded_image, prediction_result)
        
        # Add technical details
        if "Technical Details" in sections:
            add_technical_details(pdf, prediction_result)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disease_name = prediction_result['disease_name'].replace(' ', '_')
        filename = f"plant_disease_report_{disease_name}_{timestamp}.pdf"
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", filename)
        
        # Save PDF
        pdf.output(filepath)
        
        return filepath
        
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return None

def add_title_page(pdf, prediction_result):
    """Add title page to PDF"""
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 20, 'Plant Disease Detection Report', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Results', 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Disease: {prediction_result['disease_name']}", 0, 1, 'L')
    pdf.cell(0, 8, f"Plant Type: {prediction_result['plant_type']}", 0, 1, 'L')
    pdf.cell(0, 8, f"Confidence: {prediction_result['confidence']:.2f}%", 0, 1, 'L')
    pdf.cell(0, 8, f"Analysis Date: {prediction_result['timestamp']}", 0, 1, 'L')
    
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, 'Generated by Plant Disease Detection AI System', 0, 1, 'C')
    
    pdf.add_page()

def add_analysis_summary(pdf, prediction_result, disease_info):
    """Add disease analysis summary to PDF"""
    pdf.chapter_title("Disease Analysis Summary")
    
    # Prediction details
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Prediction Results:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"• Detected Disease: {prediction_result['disease_name']}", 0, 1, 'L')
    pdf.cell(0, 6, f"• Plant Type: {prediction_result['plant_type']}", 0, 1, 'L')
    pdf.cell(0, 6, f"• Confidence Level: {prediction_result['confidence']:.2f}%", 0, 1, 'L')
    pdf.cell(0, 6, f"• Model Confidence: {prediction_result['model_confidence'].title()}", 0, 1, 'L')
    pdf.ln(5)
    
    # Disease description
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Disease Description:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(disease_info.get('description', 'No description available.'))
    
    # Symptoms
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Common Symptoms:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(disease_info.get('symptoms', 'No symptoms information available.'))
    
    pdf.ln(5)

def add_treatment_recommendations(pdf, disease_info):
    """Add treatment recommendations to PDF"""
    pdf.chapter_title("Treatment Recommendations")
    
    # Immediate actions
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Immediate Actions:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(disease_info.get('immediate_actions', 'No immediate actions specified.'))
    pdf.ln(5)
    
    # Treatment time
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Expected Treatment Time:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, disease_info.get('treatment_time', 'Varies'), 0, 1, 'L')
    pdf.ln(5)
    
    # Organic treatments
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Organic Treatment Options:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    organic_treatments = disease_info.get('organic_treatments', [])
    if organic_treatments:
        for treatment in organic_treatments:
            pdf.cell(0, 6, f"• {treatment}", 0, 1, 'L')
    else:
        pdf.cell(0, 6, "No organic treatments specified.", 0, 1, 'L')
    pdf.ln(5)
    
    # Chemical treatments
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Chemical Treatment Options:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    chemical_treatments = disease_info.get('chemical_treatments', [])
    if chemical_treatments:
        for treatment in chemical_treatments:
            pdf.cell(0, 6, f"• {treatment}", 0, 1, 'L')
    else:
        pdf.cell(0, 6, "No chemical treatments specified.", 0, 1, 'L')
    
    pdf.ln(5)

def add_prevention_tips(pdf, disease_info):
    """Add prevention tips to PDF"""
    pdf.chapter_title("Prevention Strategies")
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Prevention Measures:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.chapter_body(disease_info.get('prevention', 'No prevention information available.'))
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'General Prevention Tips:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    general_tips = [
        "• Maintain proper plant spacing for good air circulation",
        "• Water plants at the base to avoid wetting foliage",
        "• Remove and destroy infected plant debris",
        "• Practice crop rotation when possible",
        "• Use disease-resistant plant varieties",
        "• Monitor plants regularly for early signs of disease",
        "• Maintain proper soil fertility and pH",
        "• Avoid working with plants when they are wet"
    ]
    
    for tip in general_tips:
        pdf.cell(0, 6, tip, 0, 1, 'L')
    
    pdf.ln(5)

def add_image_analysis(pdf, uploaded_image, prediction_result):
    """Add image analysis section to PDF"""
    pdf.chapter_title("Image Analysis")
    
    try:
        # Get image information
        image = Image.open(uploaded_image)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Image Information:', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"• Format: {image.format}", 0, 1, 'L')
        pdf.cell(0, 6, f"• Size: {image.size[0]} x {image.size[1]} pixels", 0, 1, 'L')
        pdf.cell(0, 6, f"• Mode: {image.mode}", 0, 1, 'L')
        pdf.cell(0, 6, f"• File Size: {uploaded_image.size / 1024:.1f} KB", 0, 1, 'L')
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Analysis Notes:', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.chapter_body("The image was processed using advanced computer vision techniques and analyzed by our deep learning model for plant disease detection.")
        
    except Exception as e:
        pdf.set_font('Arial', '', 10)
        pdf.chapter_body(f"Error analyzing image: {str(e)}")
    
    pdf.ln(5)

def add_technical_details(pdf, prediction_result):
    """Add technical details to PDF"""
    pdf.chapter_title("Technical Details")
    
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Model Information:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, "• Architecture: Convolutional Neural Network (CNN)", 0, 1, 'L')
    pdf.cell(0, 6, "• Base Model: VGG16 with transfer learning", 0, 1, 'L')
    pdf.cell(0, 6, "• Training Dataset: PlantVillage Dataset", 0, 1, 'L')
    pdf.cell(0, 6, "• Number of Classes: 38 disease categories", 0, 1, 'L')
    pdf.cell(0, 6, f"• Predicted Class Index: {prediction_result['class_index']}", 0, 1, 'L')
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Top 3 Predictions:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    for i, pred in enumerate(prediction_result.get('top_3_predictions', []), 1):
        pdf.cell(0, 6, f"{i}. {pred['disease']} - {pred['confidence']:.2f}%", 0, 1, 'L')
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Analysis Timestamp:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, prediction_result['timestamp'], 0, 1, 'L')

def create_simple_report(prediction_result, disease_info):
    """
    Create a simple text-based report for quick reference.
    
    Args:
        prediction_result: Dictionary containing prediction results
        disease_info: Dictionary containing disease information
        
    Returns:
        str: Simple report text
    """
    report = f"""
PLANT DISEASE DETECTION REPORT
==============================

Analysis Date: {prediction_result['timestamp']}
Disease: {prediction_result['disease_name']}
Plant Type: {prediction_result['plant_type']}
Confidence: {prediction_result['confidence']:.2f}%

DESCRIPTION:
{disease_info.get('description', 'No description available.')}

SYMPTOMS:
{disease_info.get('symptoms', 'No symptoms information available.')}

IMMEDIATE ACTIONS:
{disease_info.get('immediate_actions', 'No immediate actions specified.')}

PREVENTION:
{disease_info.get('prevention', 'No prevention information available.')}

TREATMENT TIME: {disease_info.get('treatment_time', 'Varies')}

ORGANIC TREATMENTS:
{', '.join(disease_info.get('organic_treatments', ['None specified']))}

CHEMICAL TREATMENTS:
{', '.join(disease_info.get('chemical_treatments', ['None specified']))}

---
Generated by Plant Disease Detection AI System
    """
    
    return report.strip() 