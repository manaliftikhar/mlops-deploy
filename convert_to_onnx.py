import torch
import torch.onnx
import onnx
import onnxruntime
import urllib.request
import os
from pytorch_model import Classifier, BasicBlock
import numpy as np
from PIL import Image


def download_model_weights(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading model weights from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Model weights saved to {filename}")
            with open(filename, 'rb') as f:
                first_bytes = f.read(10)
                if first_bytes.startswith(b'<'):
                    print("  Downloaded file appears to be HTML, not binary weights!")
                    print("   The Dropbox URL may be incorrect.")
                    print("   Try changing ?dl=0 to ?dl=1 in the URL")
                    os.remove(filename)
                    return False
                    
            file_size = os.path.getsize(filename)
            if file_size < 1024 * 1024:  
                print(f"Downloaded file too small: {file_size} bytes")
                print("Expected model weights to be much larger")
                os.remove(filename)
                return False
                
            print(f"Downloaded file size: {file_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"Error downloading weights: {e}")
            return False
    else:
        print(f"Model weights already exist at {filename}")
        
        # Verify existing file
        file_size = os.path.getsize(filename)
        print(f"✓ Existing file size: {file_size / (1024*1024):.1f} MB")
        
    return True


def convert_pytorch_to_onnx(pytorch_model_path, onnx_model_path):
    """Convert PyTorch model to ONNX format."""
    
    try:
        model = Classifier(BasicBlock, [2, 2, 2, 2])
        
        if os.path.exists(pytorch_model_path):
            print(f"Loading weights from {pytorch_model_path}")
            model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
        else:
            raise FileNotFoundError(f"Weight file not found: {pytorch_model_path}")
        
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        print(f"Converting PyTorch model to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model saved to {onnx_model_path}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def verify_onnx_model(onnx_model_path, pytorch_model_path):
    """Verify that ONNX model produces similar outputs to PyTorch model."""
    
    try:
        pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
        pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
        pytorch_model.eval()
        
        onnx_session = onnxruntime.InferenceSession(onnx_model_path)
        
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        
        onnx_input = {onnx_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = onnx_session.run(None, onnx_input)[0]
        
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        print(f"Max difference between PyTorch and ONNX outputs: {max_diff}")
        
        if max_diff < 1e-5:
            print("ONNX model verification successful!")
            return True
        else:
            print("ONNX model verification failed!")
            return False
            
    except Exception as e:
        print(f"Error during verification: {e}")
        return False


def main():
    """Main conversion process."""
    model_weights_url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
    pytorch_weights_path = "pytorch_model_weights.pth"
    onnx_model_path = "classifier_model.onnx"
    
    try:
        if not download_model_weights(model_weights_url, pytorch_weights_path):
            print("  Failed to download model weights!")
            return False
        
        if not convert_pytorch_to_onnx(pytorch_weights_path, onnx_model_path):
            print("  Failed to convert model to ONNX!")
            return False
        if not verify_onnx_model(onnx_model_path, pytorch_weights_path):
            print("   Warning: ONNX model verification failed, but model was created")
        
        onnx_model = onnx.load(onnx_model_path)
        print(f"\nONNX Model Info:")
        print(f"- Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"- Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
        
        print("\n Model conversion completed successfully!")
        print(f"ONNX model saved as: {onnx_model_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 