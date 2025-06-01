#!/usr/bin/env python3
"""
Test script for deployed Cerebrium model.
Tests both local model and deployed API.

IMPORTANT: This script is designed to be runnable without requesting further information.
- Defaults to LOCAL testing mode when no API credentials provided
- Can test deployed API when CEREBRIUM_API_KEY and CEREBRIUM_ENDPOINT are set
- Includes preset tests as required

Usage:
    python test_server.py --image path/to/image.jpg          # Test single image
    python test_server.py --run-tests                        # Run preset tests
    python test_server.py --api --image path/to/image.jpg    # Test deployed API
"""

import argparse
import requests
import json
import time
import os
import sys
from typing import Dict, Any, List, Tuple

try:
    from model import ClassificationPipeline
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False


class CerebriumTester:
    """Test deployed Cerebrium model and local fallback."""
    
    def __init__(self, api_key: str = None, endpoint: str = None):
        """
        Initialize tester.
        
        Args:
            api_key: Cerebrium API key (optional)
            endpoint: Cerebrium API endpoint (optional)
        """
        self.api_key = api_key or os.getenv('CEREBRIUM_API_KEY')
        self.endpoint = endpoint or os.getenv('CEREBRIUM_ENDPOINT')
        
        if not self.api_key:
            self.api_key = "your-api-key-here" 
        if not self.endpoint:
            self.endpoint = "https://api.cerebrium.ai/v1/your-app-name/predict" 
            
        self.api_available = self._check_api_availability()
        
        self.local_model = None
        if LOCAL_MODEL_AVAILABLE and os.path.exists("classifier_model.onnx"):
            try:
                self.local_model = ClassificationPipeline("classifier_model.onnx")
                print("Local model loaded as fallback")
            except Exception as e:
                print(f"Could not load local model: {e}")
    
    def _check_api_availability(self) -> bool:
        """Check if API is available and configured."""
        if not self.api_key or self.api_key == "your-api-key-here":
            return False
        if not self.endpoint or "your-app-name" in self.endpoint:
            return False
        return True
    
    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint."""
        if not self.api_available:
            return {"status": "error", "message": "API not configured"}
            
        try:
            health_url = self.endpoint.replace('/predict', '/health')
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(health_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {"status": "healthy", "response_time": response.elapsed.total_seconds()}
            else:
                return {"status": "unhealthy", "status_code": response.status_code}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def test_deployed_model(self, image_path: str) -> Dict[str, Any]:
        """Test deployed Cerebrium model."""
        if not self.api_available:
            return {"status": "error", "message": "API not configured - using local model instead"}
        
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image file not found: {image_path}"}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "image_path": image_path,
                "return_top_k": 5
            }
            
            start_time = time.time()
            response = requests.post(
                self.endpoint, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                result["api_response_time"] = response_time
                return result
            else:
                return {
                    "status": "error",
                    "message": f"API returned status {response.status_code}",
                    "response": response.text[:200]
                }
                
        except Exception as e:
            return {"status": "error", "message": f"API request failed: {str(e)}"}
    
    def test_local_model(self, image_path: str) -> Dict[str, Any]:
        """Test local model as fallback."""
        if not self.local_model:
            return {"status": "error", "message": "Local model not available"}
        
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"Image file not found: {image_path}"}
        
        try:
            start_time = time.time()
            
            predicted_class = self.local_model.classify_image(image_path)
            top_5 = self.local_model.classify_with_confidence(image_path, k=5)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "top_5_predictions": top_5,
                "inference_time": inference_time,
                "model_type": "local"
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Local model failed: {str(e)}"}
    
    def test_single_image(self, image_path: str, use_api: bool = False) -> Dict[str, Any]:
        """Test single image classification."""
        print(f"\nTesting {'deployed API' if use_api and self.api_available else 'local model'} with image: {image_path}")
        
        if use_api and self.api_available:
            result = self.test_deployed_model(image_path)
            if result["status"] != "error":
                return result
            else:
                print(f"API failed: {result['message']}")
                print("Falling back to local model...")
        
        return self.test_local_model(image_path)
    
    def run_preset_tests(self, use_api: bool = False) -> Dict[str, Any]:
        """Run preset tests on known images."""
        print("\n Running preset tests...")
        
       
        test_cases = [
            {
                "image": "n01440764_tench.jpeg",
                "expected_class": 0,
                "description": "Tench fish (should predict class 0)"
            },
            {
                "image": "n01667114_mud_turtle.JPEG", 
                "expected_class": 35,
                "description": "Mud turtle (should predict class 35)"
            }
        ]
        
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test_case in test_cases:
            print(f"\n  Testing: {test_case['description']}")
            
            if not os.path.exists(test_case["image"]):
                print(f"Image not found: {test_case['image']}")
                results["failed"] += 1
                results["test_results"].append({
                    "image": test_case["image"],
                    "status": "failed",
                    "reason": "Image file not found"
                })
                continue
            
            result = self.test_single_image(test_case["image"], use_api)
            
            if result["status"] == "success":
                predicted_class = result["predicted_class"]
                expected_class = test_case["expected_class"]
                
                is_correct = predicted_class == expected_class
                
                print(f"    Expected: Class {expected_class}")
                print(f"    Predicted: Class {predicted_class}")
                print(f"    Result: {' PASS' if is_correct else ' FAIL'}")
                print(f"    Inference time: {result.get('inference_time', 0):.3f}s")
                
                if is_correct:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["test_results"].append({
                    "image": test_case["image"],
                    "expected_class": expected_class,
                    "predicted_class": predicted_class,
                    "correct": is_correct,
                    "inference_time": result.get("inference_time", 0),
                    "status": "passed" if is_correct else "failed"
                })
            else:
                print(f"Test failed: {result['message']}")
                results["failed"] += 1
                results["test_results"].append({
                    "image": test_case["image"],
                    "status": "error",
                    "error": result["message"]
                })
        
        return results


def main():
    """Main function - runnable without requesting further information."""
    parser = argparse.ArgumentParser(
        description="Test Cerebrium deployment (defaults to local testing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python test_server.py --image n01440764_tench.jpeg     # Test single image locally
            python test_server.py --run-tests                      # Run preset tests locally  
            python test_server.py --api --image test.jpg           # Test deployed API
            python test_server.py --api --run-tests                # Test deployed API with presets
            
            API Configuration:
            Set CEREBRIUM_API_KEY and CEREBRIUM_ENDPOINT environment variables
            or the script will default to local testing mode.
                    """
    )
    
    parser.add_argument('--image', type=str, help='Path to image file to test')
    parser.add_argument('--run-tests', action='store_true', help='Run preset tests')
    parser.add_argument('--api', action='store_true', help='Test deployed API (requires API config)')
    parser.add_argument('--api-key', type=str, help='Cerebrium API key')
    parser.add_argument('--endpoint', type=str, help='Cerebrium API endpoint')
    
    args = parser.parse_args()
    
    if not any([args.image, args.run_tests]):
        print("🧪 No specific test requested - running basic validation...")
        print("Use --help for more options\n")
        args.run_tests = True  
    
    tester = CerebriumTester(api_key=args.api_key, endpoint=args.endpoint)
    
    print("🔧 Configuration Status:")
    print(f"   API Available: {'true' if tester.api_available else 'false'}")
    print(f"   Local Model: {'true' if tester.local_model else 'false'}")
    print(f"   Test Mode: {'API' if args.api and tester.api_available else 'Local'}")
    
    try:
        if args.image:
            if not os.path.exists(args.image):
                print(f" Error: Image file not found: {args.image}")
                return 1
            
            print(f"\n Image loaded successfully")
            print(f" Processing image...")
            
            result = tester.test_single_image(args.image, args.api)
            
            if result["status"] == "success":
                print(f"\nClassification Results:")
                print(f"   Predicted Class: {result['predicted_class']}")
                
                if "top_5_predictions" in result:
                    print(f"   Top 5 Classes:")
                    for i, (class_id, confidence) in enumerate(result["top_5_predictions"][:5], 1):
                        print(f"     {i}. Class {class_id} (confidence: {confidence:.1%})")
                
                print(f"Inference Time: {result.get('inference_time', 0):.3f}s")
                print(f" Test completed successfully!")
                
                inference_time = result.get('inference_time', 0)
                if inference_time > 3.0:
                    print(f" WARNING: Inference time ({inference_time:.3f}s) exceeds 3 second requirement!")
                    return 1
                    
            else:
                print(f"Test failed: {result['message']}")
                return 1
        
        if args.run_tests:
            results = tester.run_preset_tests(args.api)
            
            print(f"\nTest Summary:")
            print(f"   Total Tests: {results['total_tests']}")
            print(f"   Passed: {results['passed']} ")
            print(f"   Failed: {results['failed']} ")
            print(f"   Success Rate: {results['passed']}/{results['total_tests']} ({100*results['passed']/results['total_tests']:.1f}%)")
            
            if results['failed'] > 0:
                print(f"\n Some tests failed!")
                return 1
            else:
                print(f"\n All tests passed!")
                
        return 0
        
    except KeyboardInterrupt:
        print("\n\n Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 