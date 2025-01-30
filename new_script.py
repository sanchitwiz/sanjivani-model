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
    import sys
    if len(sys.argv) > 1:  # If image path provided in command line
        test_image(sys.argv[1])
    else:  # Otherwise start Flask app
        app.run(debug=True, port=5100)