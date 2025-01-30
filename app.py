import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
import torchvision
from torchvision import transforms
from PIL import Image
import os
import gdown

app = Flask(__name__)

os.makedirs('models', exist_ok=True)

MODEL_PATH = "models/Wound_classification.pth"
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1Ca7DO7HpRLzduqyNTf5ldOW8r_hTkD0W"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# Load pre-trained Vision Transformer model
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

# Freeze the layers
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# Customize the head for 10 output classes
torch.manual_seed(42)
pretrained_vit.heads = nn.Linear(in_features=768, out_features=10)

# Load your custom-trained weights with map_location to CPU
try:
    pretrained_vit.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

pretrained_vit.eval()

# Preprocessing function for input images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = {
    0: "Abrasions",
    1: "Bruises", 
    2: "Burns",
    3: "Cut",
    4: "Diabetic Wound",
    5: "Laceration",
    6: "Normal",
    7: "Pressure Wound",
    8: "Surgical Wound",
    9: "Venous Wound"}

@app.route('/')
def index():
    return render_template('index.html')  # The front-end HTML page to upload images

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        file = request.files['file']
        img = Image.open(file)

        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = pretrained_vit(img_tensor)
            _, predicted_class = torch.max(output, 1)

        class_idx = predicted_class.item()
        class_name = class_names[class_idx]

        # Return the predicted class index and injury name
        return jsonify({
            "predicted_class_index": class_idx,
            "predicted_injury_name": class_name
        })

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"})

# Add this new function
def test_image(image_path):
    """Test a local image and print results to terminal"""
    try:
        img = Image.open(image_path)
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = pretrained_vit(img_tensor)
            _, predicted_class = torch.max(output, 1)
        
        class_idx = predicted_class.item()
        class_name = class_names[class_idx]
        
        print("\n=== Prediction Results ===")
        print(f"Image path: {image_path}")
        print(f"Predicted class: {class_idx} ({class_name})")
        print("==========================\n")
        
    except Exception as e:
        print(f"\nError processing image: {str(e)}\n")

# Modified main block
if __name__ == '__main__':
    # Use waitress server for production
    from waitress import serve
    print("Server Running on http://localhost:5100")
    serve(app, host="0.0.0.0", port=5100)


# if __name__ == '__main__':
#     app.run(debug=True, port=5100)