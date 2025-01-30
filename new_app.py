import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import torchvision
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

torch.manual_seed(42)
pretrained_vit.heads = nn.Linear(in_features=768, out_features=10)
pretrained_vit.load_state_dict(torch.load("models/Wound_classification.pth", weights_only=True))
pretrained_vit.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file found in request"}), 400

        file = request.files['file']
        img = Image.open(file)

        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = pretrained_vit(img_tensor)
            _, predicted_class = torch.max(output, 1)

        return jsonify({"predicted_class": predicted_class.item()})

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True, port=5100)
