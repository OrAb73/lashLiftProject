import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt

# Step 1: Define the model architecture (same as in training)
def initialize_model():
    # Load ResNet-50 (without pretrained weights initially)
    model = models.resnet50(weights=None)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # 2 classes: lift rod and round rod
    )
    
    return model

# Step 2: Set up image preprocessing (same as test transforms in training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 3: Load the trained model
def load_model(model_path):
    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # print(f"Using device: {device}")
    
    # Initialize the model architecture
    model = initialize_model()
    
    # Load the saved model weights
    #print("Loading model weights...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    #print("Model loaded successfully!")
    
    return model, device

# Step 4: Function to predict on a single image
def predict_image(model, image_path, device, show_image=True):
    # Define class names
    class_names = ['lift_rod_eyes', 'round_rod_eyes']
    
    try:
        print(f"Opening image from: {image_path}")
        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        print("Image opened successfully")
        
        # Display the image if requested
        if show_image:
            plt.figure(figsize=(6, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title("Test Image")
            plt.savefig("test_image_display.png")  # Save instead of display for headless environments
            #print("Image saved as test_image_display.png")
            plt.close()
        
        # Preprocess and create tensor
        print("Preprocessing image...")
        image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Perform inference
        #print("Running inference...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)
            
            predicted_class = class_names[predicted_idx.item()]
            confidence = probs[0][predicted_idx.item()].item() * 100
            
            # Get probabilities for both classes
            class_probabilities = {
                class_names[i]: float(probs[0][i]) * 100 for i in range(len(class_names))
            }
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nClass Probabilities:")
        for cls, prob in class_probabilities.items():
            print(f"{cls}: {prob:.2f}%")
            
        return predicted_class, confidence, class_probabilities
            
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, {}

# Main execution - just change these paths to match your environment
def main():
    # Path to your model file - use raw string
    model_path = r"C:\Users\Or\Desktop\testModel\lash_rod_classifier.pth"
    
    # Path to your test image - use raw string
    test_image_path = r"C:\Users\Or\Desktop\testModel\R3.png" # R3 103711
    
    # Load the model
    model, device = load_model(model_path)
    
    # Make a prediction
    predict_image(model, test_image_path, device)

if __name__ == "__main__":
    main()