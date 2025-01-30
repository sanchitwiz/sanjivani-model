import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import requests
from torchvision import transforms
import torchvision
from PIL import Image
from io import BytesIO

# Define the Flask app
app = Flask(__name__)

# Load the pretrained model (ViT example)
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

pretrained_vit.heads = nn.Linear(in_features=768, out_features=10)
pretrained_vit.load_state_dict(torch.load("models/Wound_classification.pth", weights_only=True))
pretrained_vit.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the route that takes the image URL as input
@app.route('/predict-url', methods=['GET'])
def predict_url():
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))  # Use BytesIO to handle in-memory bytes

        # Preprocess the image
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

        # Run the model and get prediction
        with torch.no_grad():
            output = pretrained_vit(img_tensor)
            _, predicted_class = torch.max(output, 1)

        return jsonify({"predicted_class": predicted_class.item()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)