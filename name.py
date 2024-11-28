import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from deepface import DeepFace
import os
from PIL import Image
import numpy as np

# 1. Check if GPU is available for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Load and display image
image = cv2.imread("C:\\Users\\shiva\\Desktop\\picture 1.jpg")
print("Original Image Shape:", image.shape)

# Display image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# Resize the image to fit the screen if it's too large
screen_width, screen_height = 1280, 720  # Adjust this as needed
image_height, image_width = image.shape[:2]
scale_factor = min(screen_width / image_width, screen_height / image_height)

if scale_factor < 1:  # Only resize if the image is larger than the screen
    image = cv2.resize(image, (int(image_width * scale_factor), int(image_height * scale_factor)))
    print("Resized Image Shape:", image.shape)

# Display the resized image in an OpenCV window
cv2.imshow("Image", image)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# 3. Grayscale and Face Detection
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', image_gray)
cv2.waitKey(10000)
cv2.destroyAllWindows()
print("Grayscale Image Shape:", image_gray.shape)

# Load Haar Cascade Classifier for face detection
face_detector = cv2.CascadeClassifier("C:\\Users\\shiva\\Downloads\\haarcascade_frontalface_default.xml")

# Perform face detection
detections = face_detector.detectMultiScale(image_gray)
print("Detections:", detections)
print("Number of faces detected:", len(detections))

# Draw rectangles for each detected face
for (x, y, w, h) in detections:
    cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 255, 255), 5)

cv2.imshow('Detected Faces', image_gray)
cv2.waitKey(10000)
cv2.destroyAllWindows()

# 4. Load Dataset and Preprocess for Training
class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):  # Constructor accepts directory and transform
        self.directory = directory
        self.transform = transform
        self.images = []
        
        # Iterate through the directory and load images
        for filename in os.listdir(directory):
            image_path = os.path.join(directory, filename)
            if os.path.isfile(image_path):
                self.images.append(image_path)
        
    def __len__(self):  # Return number of images
        return len(self.images)
    
    def __getitem__(self, idx):  # Get image at the given index
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image


# Define transformations (resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to your dataset
dataset_path = "images"  # Change this to your dataset path

# Create the dataset and DataLoader
train_dataset = CustomImageDataset(directory=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Fine-Tune Pre-Trained ResNet50 Model
base_model = models.resnet50(pretrained=True)

# Modify the final layer to match the number of classes (in this case, using 2 classes for simplicity)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, 2)  # Adjust the number of classes based on your use case

# Move the model to the appropriate device
base_model = base_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=0.0001)

# 6. Train the model
num_epochs = 10
for epoch in range(num_epochs):
    base_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs in train_loader:
        inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = base_model(inputs)
        labels = torch.zeros(inputs.size(0), dtype=torch.long).to(device)  # Dummy labels for now
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {correct / total * 100:.2f}%")

# Save the model
torch.save(base_model.state_dict(), 'personal_face_recognition_model.pth')

# 7. Perform Face Analysis Using DeepFace (as before)
image_path = "C:\\Users\\shiva\\Desktop\\picture 1.jpg"
face_analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion', 'race'])

# Print the analysis results
print("\n--- Face Analysis Results ---\n")
if isinstance(face_analysis, list) and len(face_analysis) > 0:
    result = face_analysis[0]  # DeepFace returns a list, we'll use the first result

    print(f"Age: {result['age']}")
    print(f"Gender: {result['gender']}")
    print("Emotion Probabilities:")
    for emotion, prob in result['emotion'].items():
        print(f"  - {emotion}: {prob:.2f}%")
    
    print("Race Probabilities:")
    for race, prob in result['race'].items():
        print(f"  - {race}: {prob:.2f}%")
else:
    print("No face detected or error in analysis.")
