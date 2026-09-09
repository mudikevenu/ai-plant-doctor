TREATMENTS = {
    "Corn_(maize)___Northern_Leaf_Blight": {
        "title": "Corn Northern Leaf Blight",
        "severity": "Moderate",
        "treatment": [
            "Remove severely affected leaves when practical.",
            "Improve air circulation and avoid excessive leaf moisture.",
            "Use an appropriate fungicide according to the product label when needed.",
            "Remove infected crop debris after harvest."
        ],
        "prevention": [
            "Use resistant corn varieties when available.",
            "Practice crop rotation.",
            "Maintain good field sanitation.",
            "Monitor plants regularly, especially during warm and humid conditions."
        ]
    },

    "Apple___Apple_scab": {
        "title": "Apple Scab",
        "severity": "Moderate",
        "treatment": [
            "Remove and destroy infected leaves and fallen plant debris.",
            "Improve air circulation by pruning crowded branches.",
            "Avoid overhead watering and keep foliage dry.",
            "Use an appropriate fungicide according to the product label."
        ],
        "prevention": [
            "Keep the area around the tree clean.",
            "Prune regularly to improve airflow.",
            "Choose disease-resistant apple varieties when possible."
        ]
    },

    "Apple___Black_rot": {
        "title": "Apple Black Rot",
        "severity": "High",
        "treatment": [
            "Remove infected leaves, fruit, and dead branches.",
            "Prune affected wood well below visible symptoms.",
            "Remove mummified fruit from the tree and ground.",
            "Use an appropriate fungicide according to the product label."
        ],
        "prevention": [
            "Keep the orchard clean.",
            "Avoid unnecessary tree wounds.",
            "Maintain good air circulation."
        ]
    },

    "Apple___Cedar_apple_rust": {
        "title": "Cedar Apple Rust",
        "severity": "Moderate",
        "treatment": [
            "Remove severely affected leaves when practical.",
            "Improve air circulation around the tree.",
            "Use a suitable fungicide according to local agricultural guidance."
        ],
        "prevention": [
            "Remove nearby alternate hosts where appropriate.",
            "Keep the tree healthy with proper watering and nutrition."
        ]
    },

    "Apple___healthy": {
        "title": "Healthy Apple Leaf",
        "severity": "None",
        "treatment": [
            "No disease treatment is required.",
            "Continue normal watering, nutrition, and plant care."
        ],
        "prevention": [
            "Monitor leaves regularly for changes.",
            "Maintain good air circulation and sanitation."
        ]
    },

    "Tomato___Bacterial_spot": {
        "title": "Tomato Bacterial Spot",
        "severity": "High",
        "treatment": [
            "Remove severely infected leaves.",
            "Avoid working with plants while foliage is wet.",
            "Avoid overhead irrigation.",
            "Use disease-management products according to local agricultural guidance."
        ],
        "prevention": [
            "Use clean seeds and healthy transplants.",
            "Provide good spacing and airflow.",
            "Remove infected plant debris."
        ]
    },

    "Tomato___Early_blight": {
        "title": "Tomato Early Blight",
        "severity": "Moderate",
        "treatment": [
            "Remove infected lower leaves.",
            "Keep foliage dry and avoid overhead watering.",
            "Apply a suitable fungicide according to the product label if needed.",
            "Remove infected plant debris after harvest."
        ],
        "prevention": [
            "Use crop rotation.",
            "Mulch around plants to reduce soil splash.",
            "Maintain good spacing and airflow."
        ]
    },

    "Tomato___Late_blight": {
        "title": "Tomato Late Blight",
        "severity": "High",
        "treatment": [
            "Remove and dispose of severely infected plant material.",
            "Avoid overhead watering.",
            "Improve airflow around plants.",
            "Use an appropriate fungicide according to local agricultural guidance."
        ],
        "prevention": [
            "Avoid planting infected transplants.",
            "Monitor plants frequently during cool, wet weather.",
            "Remove infected debris promptly."
        ]
    },

    "Tomato___healthy": {
        "title": "Healthy Tomato Leaf",
        "severity": "None",
        "treatment": [
            "No disease treatment is required.",
            "Continue normal plant care."
        ],
        "prevention": [
            "Monitor the plant regularly.",
            "Maintain good spacing and airflow.",
            "Water at the base of the plant."
        ]
    },

    "Potato___Early_blight": {
        "title": "Potato Early Blight",
        "severity": "Moderate",
        "treatment": [
            "Remove severely affected foliage.",
            "Avoid overhead irrigation.",
            "Maintain adequate plant nutrition.",
            "Use an appropriate fungicide according to the product label."
        ],
        "prevention": [
            "Practice crop rotation.",
            "Remove infected crop debris.",
            "Maintain good field sanitation."
        ]
    },

    "Potato___Late_blight": {
        "title": "Potato Late Blight",
        "severity": "High",
        "treatment": [
            "Remove severely infected foliage.",
            "Avoid overhead irrigation.",
            "Use suitable disease-management products according to local agricultural guidance.",
            "Remove infected plant debris."
        ],
        "prevention": [
            "Use healthy seed potatoes.",
            "Monitor during cool and wet conditions.",
            "Practice crop rotation where appropriate."
        ]
    },

    "Grape___Black_rot": {
        "title": "Grape Black Rot",
        "severity": "High",
        "treatment": [
            "Remove infected leaves and fruit.",
            "Remove dried or mummified berries.",
            "Improve canopy airflow through appropriate pruning.",
            "Use a suitable fungicide according to the product label."
        ],
        "prevention": [
            "Keep vineyard sanitation high.",
            "Improve sunlight and airflow.",
            "Remove infected plant material."
        ]
    },

    "Grape___healthy": {
        "title": "Healthy Grape Leaf",
        "severity": "None",
        "treatment": [
            "No disease treatment is required.",
            "Continue normal vineyard management."
        ],
        "prevention": [
            "Monitor leaves regularly.",
            "Maintain good canopy airflow."
        ]
    },

    "Peach___Bacterial_spot": {
        "title": "Peach Bacterial Spot",
        "severity": "High",
        "treatment": [
            "Remove severely affected plant material.",
            "Avoid unnecessary pruning wounds.",
            "Maintain good airflow.",
            "Follow local agricultural recommendations for disease management."
        ],
        "prevention": [
            "Use healthy planting material.",
            "Avoid overhead irrigation.",
            "Maintain orchard sanitation."
        ]
    },

    "Peach___healthy": {
        "title": "Healthy Peach Leaf",
        "severity": "None",
        "treatment": [
            "No disease treatment is required.",
            "Continue normal plant care."
        ],
        "prevention": [
            "Monitor the tree regularly.",
            "Maintain good airflow and sanitation."
        ]
    }
}


DEFAULT_TREATMENT = {
    "title": "Plant Disease",
    "severity": "Unknown",
    "treatment": [
        "Remove severely affected plant material when appropriate.",
        "Improve airflow around the plant.",
        "Avoid unnecessary moisture on leaves.",
        "Consult a local agricultural expert for a confirmed diagnosis and treatment."
    ],
    "prevention": [
        "Monitor the plant regularly.",
        "Keep the growing area clean.",
        "Use healthy planting material.",
        "Follow local agricultural recommendations."
    ]
}


def get_treatment(class_name):
    return TREATMENTS.get(class_name, DEFAULT_TREATMENT)
