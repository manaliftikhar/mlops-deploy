#!/usr/bin/env python3
"""
Comprehensive test suite for the classification model.
Tests everything expected for ML model deployment.
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image
import time
from model import ImagePreprocessor, ONNXClassifier, ClassificationPipeline
from convert_to_onnx import convert_pytorch_to_onnx, download_model_weights


class TestImagePreprocessor:
    """Test image preprocessing functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.preprocessor = ImagePreprocessor()
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        assert self.preprocessor.target_size == (224, 224)
        assert np.array_equal(self.preprocessor.mean, np.array([0.485, 0.456, 0.406]))
        assert np.array_equal(self.preprocessor.std, np.array([0.229, 0.224, 0.225]))
    
    def test_preprocess_image_from_path(self):
        """Test preprocessing from image path."""
        if os.path.exists("n01440764_tench.jpeg"):
            result = self.preprocessor.preprocess_image("n01440764_tench.jpeg")
            assert result.shape == (1, 3, 224, 224)
            assert result.dtype == np.float32
        else:
            pytest.skip("Test image not found")
    
    def test_preprocess_image_from_pil(self):
        """Test preprocessing from PIL Image."""
        dummy_image = Image.new('RGB', (300, 400), color='red')
        result = self.preprocessor.preprocess_image(dummy_image)
        
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32
    
    def test_preprocess_image_from_numpy(self):
        """Test preprocessing from numpy array."""
        dummy_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess_image(dummy_array)
        
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32
    
    def test_preprocess_grayscale_to_rgb(self):
        """Test that grayscale images are converted to RGB."""
        dummy_image = Image.new('L', (224, 224), color=128)
        result = self.preprocessor.preprocess_image(dummy_image)
        
        assert result.shape == (1, 3, 224, 224)
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        if os.path.exists("n01440764_tench.jpeg") and os.path.exists("n01667114_mud_turtle.JPEG"):
            image_paths = ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]
            result = self.preprocessor.preprocess_batch(image_paths)
            
            assert result.shape == (2, 3, 224, 224)
            assert result.dtype == np.float32
        else:
            pytest.skip("Test images not found")
    
    def test_preprocess_invalid_input(self):
        """Test preprocessing with invalid input."""
        with pytest.raises(ValueError):
            self.preprocessor.preprocess_image(12345)
    
    def test_preprocess_nonexistent_file(self):
        """Test preprocessing with non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.preprocessor.preprocess_image("nonexistent_file.jpg")
    
    def test_normalization_values(self):
        """Test that normalization produces expected range."""
        white_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        result = self.preprocessor.preprocess_image(white_image)
        
        expected_normalized = (1.0 - self.preprocessor.mean) / self.preprocessor.std
        
        for i in range(3):
            channel_mean = np.mean(result[0, i, :, :])
            assert abs(channel_mean - expected_normalized[i]) < 0.1


class TestONNXClassifier:
    """Test ONNX classifier functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.model_path = "classifier_model.onnx"
        if os.path.exists(self.model_path):
            self.classifier = ONNXClassifier(self.model_path)
        else:
            pytest.skip("ONNX model not found. Run convert_to_onnx.py first.")
    
    def test_model_loading(self):
        """Test that model loads correctly."""
        assert self.classifier.session is not None
        assert self.classifier.input_name is not None
        assert self.classifier.output_name is not None
    
    def test_predict(self):
        """Test prediction functionality."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = self.classifier.predict(dummy_input)
        
        assert result.shape == (1, 1000)
        assert result.dtype == np.float32
    
    def test_predict_class(self):
        """Test class prediction."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = self.classifier.predict_class(dummy_input)
        
        assert isinstance(result, int)
        assert 0 <= result < 1000
    
    def test_predict_top_k(self):
        """Test top-k predictions."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = self.classifier.predict_top_k(dummy_input, k=5)
        
        assert len(result) == 5
        assert all(isinstance(class_id, int) for class_id, _ in result)
        assert all(isinstance(confidence, float) for _, confidence in result)
        assert all(0 <= class_id < 1000 for class_id, _ in result)
        
        confidences = [conf for _, conf in result]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_invalid_model_path(self):
        """Test loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            ONNXClassifier("nonexistent_model.onnx")
    
    def test_prediction_consistency(self):
        """Test that same input produces same output."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        result1 = self.classifier.predict(dummy_input)
        result2 = self.classifier.predict(dummy_input)
        
        np.testing.assert_array_equal(result1, result2)


class TestClassificationPipeline:
    """Test the complete classification pipeline."""
    
    def setup_method(self):
        """Setup for each test method."""
        if os.path.exists("classifier_model.onnx"):
            self.pipeline = ClassificationPipeline("classifier_model.onnx")
        else:
            pytest.skip("ONNX model not found. Run convert_to_onnx.py first.")
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        assert self.pipeline.preprocessor is not None
        assert self.pipeline.classifier is not None
    
    def test_classify_image(self):
        """Test image classification."""
        if os.path.exists("n01440764_tench.jpeg"):
            result = self.pipeline.classify_image("n01440764_tench.jpeg")
            assert isinstance(result, int)
            assert 0 <= result < 1000
        else:
            pytest.skip("Test image not found")
    
    def test_classify_with_confidence(self):
        """Test classification with confidence scores."""
        if os.path.exists("n01440764_tench.jpeg"):
            result = self.pipeline.classify_with_confidence("n01440764_tench.jpeg", k=3)
            assert len(result) == 3
            assert all(isinstance(class_id, int) for class_id, _ in result)
            assert all(isinstance(confidence, float) for _, confidence in result)
        else:
            pytest.skip("Test image not found")
    
    def test_expected_classifications(self):
        """Test that model predicts expected classes for test images."""
        test_cases = [
            ("n01440764_tench.jpeg", 0),
            ("n01667114_mud_turtle.JPEG", 35)
        ]
        
        for image_path, expected_class in test_cases:
            if os.path.exists(image_path):
                predicted_class = self.pipeline.classify_image(image_path)
                top_5 = self.pipeline.classify_with_confidence(image_path, k=5)
                top_5_classes = [class_id for class_id, _ in top_5]
                
                print(f"Image: {image_path}")
                print(f"Expected: {expected_class}, Predicted: {predicted_class}")
                print(f"Top 5: {top_5_classes}")
                
                assert predicted_class == expected_class or expected_class in top_5_classes
            else:
                pytest.skip(f"Test image {image_path} not found")


class TestPerformance:
    """Test performance requirements."""
    
    def setup_method(self):
        """Setup for performance tests."""
        if os.path.exists("classifier_model.onnx"):
            self.pipeline = ClassificationPipeline("classifier_model.onnx")
        else:
            pytest.skip("ONNX model not found. Run convert_to_onnx.py first.")
    
    def test_inference_speed(self):
        """Test that inference meets speed requirements (2-3 seconds)."""
        if not os.path.exists("n01440764_tench.jpeg"):
            pytest.skip("Test image not found")
        
        self.pipeline.classify_image("n01440764_tench.jpeg")
        
        start_time = time.time()
        result = self.pipeline.classify_image("n01440764_tench.jpeg")
        end_time = time.time()
        
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.3f} seconds")
        
        assert inference_time < 3.0
        
        if inference_time > 1.0:
            print(f"Warning: Inference time ({inference_time:.3f}s) is above 1 second")
    
    def test_batch_processing_speed(self):
        """Test batch processing performance."""
        test_images = ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if len(existing_images) < 2:
            pytest.skip("Not enough test images found")
        
        start_time = time.time()
        for image_path in existing_images:
            self.pipeline.classify_image(image_path)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(existing_images)
        
        print(f"Average time per image: {avg_time_per_image:.3f} seconds")
        assert avg_time_per_image < 3.0


class TestRobustness:
    """Test model robustness and edge cases."""
    
    def setup_method(self):
        """Setup for robustness tests."""
        if os.path.exists("classifier_model.onnx"):
            self.pipeline = ClassificationPipeline("classifier_model.onnx")
        else:
            pytest.skip("ONNX model not found. Run convert_to_onnx.py first.")
    
    def test_different_image_sizes(self):
        """Test with images of different sizes."""
        sizes = [(100, 100), (500, 300), (224, 224), (1024, 768)]
        
        for width, height in sizes:
            dummy_image = Image.new('RGB', (width, height), color='blue')
            result = self.pipeline.classify_image(dummy_image)
            assert isinstance(result, int)
            assert 0 <= result < 1000
    
    def test_extreme_image_content(self):
        """Test with extreme image content."""
        black_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        result1 = self.pipeline.classify_image(black_image)
        
        white_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        result2 = self.pipeline.classify_image(white_image)
        
        noise_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        noise_image = Image.fromarray(noise_array)
        result3 = self.pipeline.classify_image(noise_image)
        
        for result in [result1, result2, result3]:
            assert isinstance(result, int)
            assert 0 <= result < 1000


def run_all_tests():
    """Run all tests and return results."""
    print("Running comprehensive model tests...\n")
    
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
    
    exit(0 if success else 1) 