import os
import base64
import io
from PIL import Image
import numpy as np
from model import ClassificationPipeline
import traceback


print("Loading ONNX model...")
pipeline = ClassificationPipeline("classifier_model.onnx")
print("Model loaded successfully!")

def predict(image_data, return_top_k=5):
    """
    Main prediction function for Cerebrium.
    
    Args:
        image_data: Base64 encoded image or image path
        return_top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    try:
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            if len(image_data) > 500: 
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                if not os.path.exists(image_data):
                    return {"error": f"Image file not found: {image_data}"}
                image = image_data
        else:
            return {"error": "Invalid image data format"}
        
        predicted_class = pipeline.classify_image(image)
        top_k_predictions = pipeline.classify_with_confidence(image, k=return_top_k)
        
        return {
            "predicted_class": predicted_class,
            "top_predictions": [
                {"class_id": class_id, "confidence": confidence}
                for class_id, confidence in top_k_predictions
            ],
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": error_msg,
            "status": "error"
        }

def health_check():
    """Health check endpoint."""
    try:
        dummy_image = Image.new('RGB', (224, 224), color='red')
        result = pipeline.classify_image(dummy_image)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_prediction": result
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    print("Testing prediction function...")
    
    test_images = ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting with {img_path}:")
            result = predict(img_path)
            print(f"Result: {result}")
        else:
            print(f"Test image {img_path} not found")
    
    print("\nTesting health check:")
    health_result = health_check()
    print(f"Health check result: {health_result}") 