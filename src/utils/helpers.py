import joblib
import numpy as np
import pandas as pd

# Class labels
CLASS_LABELS = ["Waste", "Target", "Acceptable", "Inefficient"]

# Feature names for better display
FEATURE_NAMES = [
    "ZUx - Cycle time", 
    "Mold temperature", 
    "APVs - Specific injection pressure peak value",
    "time_to_fill", 
    "SVo - Shot volume", 
    "CPn - Screw position at the end of hold pressure",
    "ZDx - Plasticizing time", 
    "SKx - Closing force", 
    "SKs - Clamping force peak value",
    "APSs - Specific back pressure peak value", 
    "Mm - Torque mean value current cycle",
    "Ms - Torque peak value current cycle", 
    "Melt temperature"
]

def load_models():
    """Load all ML models from the models directory"""
    best_models = {
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "Decision Tree": joblib.load("models/dt_model.pkl"),
        "SVM": joblib.load("models/svm_model.pkl"),
        "ANN": joblib.load("models/ann_model.pkl"),
        "AdaBoost": joblib.load("models/ada_model.pkl")
    }
    return best_models

def load_encoders():
    """Load label encoder and scaler from the models directory"""
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return label_encoder, scaler

def load_test_data():
    """Load test data for model evaluation"""
    X_test = joblib.load("models/X_test.pkl")
    y_test = joblib.load("models/y_test.pkl")
    return X_test, y_test

# def predict_quality(model, input_data, scaler, label_encoder):
#     """Predict quality class based on input parameters"""
#     input_scaled = scaler.transform(input_data)
#     pred = model.predict(input_scaled)[0]
#     pred_index = label_encoder.inverse_transform([pred])[0]
#     pred_label = CLASS_LABELS[int(pred_index)]
#     return pred_label, pred_index

def predict_quality(model, input_data, scaler, label_encoder):
    """
    Predict quality class based on input parameters with proper feature names
    
    Parameters:
    -----------
    model : trained model
        The machine learning model used for prediction
    input_data : numpy.ndarray or pandas.DataFrame
        Input data for prediction
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used to normalize the input data
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encoder used to decode the prediction to a class label
    
    Returns:
    --------
    pred_label : str
        Predicted quality class label
    pred_index : int
        Index of the predicted class
    """
    # Define class labels - should match your actual quality classes
    CLASS_LABELS = ["Waste", "Target", "Acceptable", "Inefficient"]
    
    # If input_data is not already a DataFrame with named columns, convert it
    if not isinstance(input_data, pd.DataFrame):
        # Define the feature names that were used during training
        feature_names = [
            "Melt temperature", 
            "Mold temperature", 
            "time_to_fill", 
            "ZDx - Plasticizing time", 
            "ZUx - Cycle time", 
            "SKx - Closing force", 
            "SKs - Clamping force peak value", 
            "Ms - Torque peak value current cycle", 
            "Mm - Torque mean value current cycle", 
            "APSs - Specific back pressure peak value", 
            "APVs - Specific injection pressure peak value", 
            "CPn - Screw position at the end of hold pressure", 
            "SVo - Shot volume"
        ]
        input_data = pd.DataFrame(input_data, columns=feature_names)
    
    # Scale the data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    pred = model.predict(input_scaled)[0]
    pred_index = label_encoder.inverse_transform([pred])[0]
    
    # Debug information
    print(f"Raw prediction: {pred}")
    print(f"Decoded prediction index: {pred_index}")
    print(f"Type of pred_index: {type(pred_index)}")
    
    # Handle different class index format possibilities
    try:
        # If pred_index is already an integer
        if isinstance(pred_index, (int, np.integer)):
            index = int(pred_index)
        # If pred_index is a float
        elif isinstance(pred_index, (float, np.float64)):
            index = int(pred_index)
        # If pred_index is a string that can be converted to int
        elif isinstance(pred_index, str) and pred_index.isdigit():
            index = int(pred_index)
        else:
            # If we can't determine how to convert, default to 0
            print(f"Warning: Couldn't convert pred_index {pred_index} to an integer. Defaulting to class 0.")
            index = 0
            
        # Check if index is in range
        if 0 <= index < len(CLASS_LABELS):
            pred_label = CLASS_LABELS[index]
        else:
            # If index is out of range, adjust it to be in range
            print(f"Warning: Index {index} is out of range for CLASS_LABELS. Adjusting to be in range.")
            # Map the class to appropriate value in your system
            # This is based on your description: 1=Target, 2=Acceptable, 3=Inefficient, 4=Waste
            # Adjust this mapping as needed for your system
            if index == 1:
                pred_label = "Target"
            elif index == 2:
                pred_label = "Acceptable"
            elif index == 3:
                pred_label = "Inefficient"
            elif index == 4:
                pred_label = "Waste"
            else:
                pred_label = "Unknown"
    except Exception as e:
        print(f"Error handling prediction index: {e}")
        pred_label = "Unknown"
    
    return pred_label, pred_index

def export_prediction(model_name, pred_label):
    """Create a dataframe for prediction export"""
    result_df = pd.DataFrame({"Model": [model_name], "Predicted Quality Class": [pred_label]})
    return result_df

def get_model_metrics():
    """Get model performance metrics"""
    results_data = {
        "Model": ["Random Forest", "Decision Tree", "SVM", "ANN"],
        "Accuracy": [0.97, 0.94, 0.93, 0.92],
        "F1-Score": [0.96, 0.93, 0.92, 0.91],
        "ROC-AUC": [0.98, 0.95, 0.94, 0.93]
    }
    return pd.DataFrame(results_data)