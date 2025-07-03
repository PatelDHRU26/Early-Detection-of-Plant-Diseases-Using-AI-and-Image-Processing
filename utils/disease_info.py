"""
Disease information utility for plant disease detection.
Provides detailed information about plant diseases and treatment recommendations.
"""

# Disease information database
DISEASE_DATABASE = {
    # Tomato Diseases
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot is a common disease of tomato caused by Xanthomonas campestris pv. vesicatoria.',
        'symptoms': 'Small, dark, water-soaked lesions on leaves, stems, and fruits. Lesions may have yellow halos.',
        'immediate_actions': 'Remove infected plants, avoid overhead irrigation, apply copper-based fungicides.',
        'prevention': 'Use disease-resistant varieties, practice crop rotation, maintain proper spacing.',
        'severity': 'High',
        'treatment_time': '2-3 weeks',
        'organic_treatments': ['Copper sulfate', 'Bacillus subtilis', 'Neem oil'],
        'chemical_treatments': ['Copper fungicides', 'Streptomycin', 'Oxytetracycline']
    },
    
    'Tomato___Early_blight': {
        'description': 'Early blight is caused by the fungus Alternaria solani and affects tomato plants worldwide.',
        'symptoms': 'Dark brown spots with concentric rings on lower leaves, yellowing and defoliation.',
        'immediate_actions': 'Remove infected leaves, improve air circulation, apply fungicides.',
        'prevention': 'Mulch around plants, avoid overhead watering, remove plant debris.',
        'severity': 'Medium',
        'treatment_time': '1-2 weeks',
        'organic_treatments': ['Baking soda spray', 'Milk spray', 'Garlic extract'],
        'chemical_treatments': ['Chlorothalonil', 'Mancozeb', 'Azoxystrobin']
    },
    
    'Tomato___Late_blight': {
        'description': 'Late blight is a devastating disease caused by Phytophthora infestans.',
        'symptoms': 'Water-soaked lesions on leaves, white fungal growth, rapid plant death.',
        'immediate_actions': 'Remove infected plants immediately, apply fungicides, improve drainage.',
        'prevention': 'Use resistant varieties, avoid overhead irrigation, monitor weather conditions.',
        'severity': 'Very High',
        'treatment_time': 'Immediate action required',
        'organic_treatments': ['Copper sulfate', 'Bacillus subtilis'],
        'chemical_treatments': ['Chlorothalonil', 'Mancozeb', 'Famoxadone']
    },
    
    'Tomato___healthy': {
        'description': 'The tomato plant appears to be healthy with no visible signs of disease.',
        'symptoms': 'No symptoms detected. Plant shows normal growth and development.',
        'immediate_actions': 'Continue current care routine, monitor for any changes.',
        'prevention': 'Maintain good cultural practices, regular monitoring, proper nutrition.',
        'severity': 'None',
        'treatment_time': 'Not applicable',
        'organic_treatments': ['Regular monitoring', 'Proper nutrition'],
        'chemical_treatments': ['None required']
    },
    
    # Potato Diseases
    'Potato___Early_blight': {
        'description': 'Early blight in potatoes is caused by Alternaria solani and affects leaves and tubers.',
        'symptoms': 'Dark brown spots with target-like rings on leaves, premature defoliation.',
        'immediate_actions': 'Remove infected foliage, apply fungicides, improve air circulation.',
        'prevention': 'Use certified seed, practice crop rotation, maintain proper spacing.',
        'severity': 'Medium',
        'treatment_time': '1-2 weeks',
        'organic_treatments': ['Baking soda spray', 'Neem oil', 'Garlic extract'],
        'chemical_treatments': ['Chlorothalonil', 'Mancozeb', 'Azoxystrobin']
    },
    
    'Potato___Late_blight': {
        'description': 'Late blight is the most serious disease of potatoes, caused by Phytophthora infestans.',
        'symptoms': 'Water-soaked lesions, white fungal growth, rapid spread in cool, wet conditions.',
        'immediate_actions': 'Remove infected plants immediately, apply fungicides, improve drainage.',
        'prevention': 'Use resistant varieties, avoid overhead irrigation, monitor weather.',
        'severity': 'Very High',
        'treatment_time': 'Immediate action required',
        'organic_treatments': ['Copper sulfate', 'Bacillus subtilis'],
        'chemical_treatments': ['Chlorothalonil', 'Mancozeb', 'Famoxadone']
    },
    
    'Potato___healthy': {
        'description': 'The potato plant appears to be healthy with no visible signs of disease.',
        'symptoms': 'No symptoms detected. Plant shows normal growth and development.',
        'immediate_actions': 'Continue current care routine, monitor for any changes.',
        'prevention': 'Maintain good cultural practices, regular monitoring, proper nutrition.',
        'severity': 'None',
        'treatment_time': 'Not applicable',
        'organic_treatments': ['Regular monitoring', 'Proper nutrition'],
        'chemical_treatments': ['None required']
    },
    
    # Apple Diseases
    'Apple___Apple_scab': {
        'description': 'Apple scab is caused by Venturia inaequalis and is the most important disease of apples.',
        'symptoms': 'Olive-green to brown spots on leaves and fruits, distorted growth.',
        'immediate_actions': 'Remove infected leaves and fruits, apply fungicides, improve air circulation.',
        'prevention': 'Use resistant varieties, remove fallen leaves, maintain tree health.',
        'severity': 'High',
        'treatment_time': '2-3 weeks',
        'organic_treatments': ['Sulfur', 'Bacillus subtilis', 'Neem oil'],
        'chemical_treatments': ['Captan', 'Myclobutanil', 'Tebuconazole']
    },
    
    'Apple___Black_rot': {
        'description': 'Black rot is caused by Botryosphaeria obtusa and affects leaves, fruits, and branches.',
        'symptoms': 'Purple spots on leaves, black rot on fruits, cankers on branches.',
        'immediate_actions': 'Prune infected branches, remove mummified fruits, apply fungicides.',
        'prevention': 'Prune for good air circulation, remove infected plant parts, maintain tree health.',
        'severity': 'Medium',
        'treatment_time': '2-4 weeks',
        'organic_treatments': ['Copper sulfate', 'Bacillus subtilis'],
        'chemical_treatments': ['Captan', 'Thiophanate-methyl', 'Tebuconazole']
    },
    
    'Apple___healthy': {
        'description': 'The apple tree appears to be healthy with no visible signs of disease.',
        'symptoms': 'No symptoms detected. Tree shows normal growth and development.',
        'immediate_actions': 'Continue current care routine, monitor for any changes.',
        'prevention': 'Maintain good cultural practices, regular monitoring, proper nutrition.',
        'severity': 'None',
        'treatment_time': 'Not applicable',
        'organic_treatments': ['Regular monitoring', 'Proper nutrition'],
        'chemical_treatments': ['None required']
    },
    
    # Corn Diseases
    'Corn___Common_rust': {
        'description': 'Common rust is caused by Puccinia sorghi and affects corn leaves and stalks.',
        'symptoms': 'Reddish-brown pustules on leaves, reduced photosynthesis, yield loss.',
        'immediate_actions': 'Apply fungicides, remove infected plant debris, improve air circulation.',
        'prevention': 'Use resistant hybrids, avoid late planting, maintain proper spacing.',
        'severity': 'Medium',
        'treatment_time': '1-2 weeks',
        'organic_treatments': ['Sulfur', 'Bacillus subtilis'],
        'chemical_treatments': ['Azoxystrobin', 'Pyraclostrobin', 'Tebuconazole']
    },
    
    'Corn___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight is caused by Exserohilum turcicum and affects corn leaves.',
        'symptoms': 'Elliptical, tan to brown lesions on leaves, reduced photosynthesis.',
        'immediate_actions': 'Apply fungicides, remove infected debris, improve air circulation.',
        'prevention': 'Use resistant hybrids, practice crop rotation, maintain proper spacing.',
        'severity': 'Medium',
        'treatment_time': '1-2 weeks',
        'organic_treatments': ['Bacillus subtilis', 'Neem oil'],
        'chemical_treatments': ['Azoxystrobin', 'Pyraclostrobin', 'Tebuconazole']
    },
    
    'Corn___healthy': {
        'description': 'The corn plant appears to be healthy with no visible signs of disease.',
        'symptoms': 'No symptoms detected. Plant shows normal growth and development.',
        'immediate_actions': 'Continue current care routine, monitor for any changes.',
        'prevention': 'Maintain good cultural practices, regular monitoring, proper nutrition.',
        'severity': 'None',
        'treatment_time': 'Not applicable',
        'organic_treatments': ['Regular monitoring', 'Proper nutrition'],
        'chemical_treatments': ['None required']
    }
}

def get_disease_info(disease_name):
    """
    Get detailed information about a specific disease.
    
    Args:
        disease_name: Name of the disease
        
    Returns:
        dict: Disease information
    """
    # Try to find exact match
    if disease_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[disease_name]
    
    # Try to find partial match
    for key, value in DISEASE_DATABASE.items():
        if disease_name.lower() in key.lower() or key.lower() in disease_name.lower():
            return value
    
    # Return default information if not found
    return {
        'description': f'Information about {disease_name} is not available in our database.',
        'symptoms': 'Symptoms may vary. Please consult with a plant disease expert.',
        'immediate_actions': 'Remove infected plant parts and improve growing conditions.',
        'prevention': 'Practice good cultural practices and monitor plants regularly.',
        'severity': 'Unknown',
        'treatment_time': 'Varies',
        'organic_treatments': ['General plant care', 'Proper nutrition'],
        'chemical_treatments': ['Consult expert for specific recommendations']
    }

def get_treatment_recommendations(disease_name, severity='medium'):
    """
    Get treatment recommendations based on disease and severity.
    
    Args:
        disease_name: Name of the disease
        severity: Severity level (low, medium, high, very high)
        
    Returns:
        dict: Treatment recommendations
    """
    disease_info = get_disease_info(disease_name)
    
    recommendations = {
        'immediate': [],
        'short_term': [],
        'long_term': [],
        'prevention': []
    }
    
    # Immediate actions
    if severity in ['high', 'very high']:
        recommendations['immediate'].extend([
            'Remove infected plants immediately',
            'Isolate affected area',
            'Apply appropriate fungicides',
            'Improve air circulation'
        ])
    else:
        recommendations['immediate'].extend([
            'Remove infected plant parts',
            'Apply treatment as recommended',
            'Monitor for spread'
        ])
    
    # Short-term actions
    recommendations['short_term'].extend([
        'Continue treatment for recommended duration',
        'Monitor plant response',
        'Adjust treatment if necessary'
    ])
    
    # Long-term actions
    recommendations['long_term'].extend([
        'Implement prevention strategies',
        'Consider resistant varieties',
        'Improve overall plant health'
    ])
    
    # Prevention
    recommendations['prevention'].extend([
        disease_info.get('prevention', 'Practice good cultural practices'),
        'Regular monitoring',
        'Proper nutrition and watering',
        'Crop rotation where applicable'
    ])
    
    return recommendations

def get_severity_level(disease_name):
    """
    Get the severity level of a disease.
    
    Args:
        disease_name: Name of the disease
        
    Returns:
        str: Severity level
    """
    disease_info = get_disease_info(disease_name)
    return disease_info.get('severity', 'Unknown')

def get_all_diseases():
    """
    Get a list of all diseases in the database.
    
    Returns:
        list: List of disease names
    """
    return list(DISEASE_DATABASE.keys())

def search_diseases(query):
    """
    Search for diseases based on a query.
    
    Args:
        query: Search query
        
    Returns:
        list: List of matching diseases
    """
    query = query.lower()
    matches = []
    
    for disease_name in DISEASE_DATABASE.keys():
        if query in disease_name.lower():
            matches.append(disease_name)
    
    return matches 