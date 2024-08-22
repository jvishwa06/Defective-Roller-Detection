import torch
import numpy as np
from scipy.stats import weibull_min
from scipy.special import softmax
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the correct path to your YOLOv8 model

# Function to predict using YOLOv8
def predict(image_path):
    # Load the image
    image = Image.open(image_path)
    # Perform prediction
    results = model(image)
    # Extract predictions from results
    if len(results) > 0:
        result = results[0]
        # Extract confidence scores
        if result.probs is not None:
            scores = result.probs.numpy()  # Use probs to get class probabilities
            if len(scores) > 0:
                scores = np.max(scores, axis=1)  # Get maximum probability for each detection
            else:
                scores = np.array([])
        else:
            scores = np.array([])
    else:
        scores = np.array([])  # Empty if no results
    return scores, results

# Function to fit a Weibull distribution to the distance data
def fit_weibull(data):
    shape, loc, scale = weibull_min.fit(data, floc=0)
    return shape, scale

# Function to compute OpenMax probabilities
def compute_openmax(scores, mav, weibull_model):
    if len(scores) == 0:
        return np.array([])  # Handle empty scores

    # Compute distance between the scores and the MAV
    distance = np.linalg.norm(scores - mav)

    # Weibull CDF to calculate the "unknown" class score
    weibull_cdf = weibull_min.cdf(distance, *weibull_model)
    adjusted_scores = scores * (1 - weibull_cdf)

    # Calculate the "unknown" class probability
    unknown_prob = np.sum(scores * weibull_cdf)
    openmax_scores = np.append(adjusted_scores, unknown_prob)

    # Apply softmax to get final probabilities
    final_probs = softmax(openmax_scores)
    return final_probs

# Example usage
def detect_with_openmax(image_path):
    # Predict using YOLOv8
    scores, results = predict(image_path)
    
    # Save the image
    image = Image.open(image_path)
    image.save('processed_image.jpg')

    # Debugging information
    print(f"Scores: {scores}")
    print(f"Results: {results}")

    # Simulated example for Mean Activation Vector (MAV) and scores
    # These would typically be computed during training
    mav = np.array([0.2, 0.3, 0.4])  # Replace with actual MAV from training

    # Check if scores are empty
    if len(scores) == 0:
        print("No objects detected")
        return

    # Fit Weibull distribution to the distance data
    distances = np.array([0.1, 0.15, 0.2, 0.05, 0.12])  # Example distances from training data
    weibull_model = fit_weibull(distances)

    # Compute OpenMax probabilities
    openmax_probs = compute_openmax(scores, mav, weibull_model)

    # Determine the class with the highest probability
    predicted_class = np.argmax(openmax_probs)
    if predicted_class == len(mav):  # If the highest prob is the "unknown" class
        print("Detected unknown object")
    else:
        print(f"Detected known class: {predicted_class} with probability: {openmax_probs[predicted_class]}")

# Running the detection on an example image
image_path = 'man.jpeg'  # Replace with the actual path to your image
detect_with_openmax(image_path)
