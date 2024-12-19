
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'saved-model_MPIIy1.keras'  # Path to your saved model
TEST_DIR = "C:\\Users\\saivi\\Desktop\\SEM-6\\DL\\Pose-estimation-using-CNN\\test\\test\\test"  # Directory containing your test images
OUTPUT_DIR = "C:\\Users\\saivi\\Desktop\\SEM-6\\DL\\Pose-estimation-using-CNN\\testresult4"  # Directory to save results
IMG_WIDTH = 220
IMG_HEIGHT = 220

# --- Functions ---
def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
    ]
    return limbs

def draw_skeleton(image, coordinates):
    """Draws the skeleton on the image based on predicted coordinates."""

    # Assuming the model outputs a tensor of shape (1, 32)
    x = coordinates[0, 0::2]  # Extract x-coordinates (every other element)
    y = coordinates[0, 1::2]  # Extract y-coordinates (every other element)

    # Transform keypoints to original image scale
    x_trans = x * image.shape[1]
    y_trans = y * image.shape[0]

    for i, (start, end) in enumerate(KEYPOINT_CONNECTIONS):
         # Make sure start and end indices are within the bounds of the array
        if start < 16 and end < 16:  
            cv2.line(image, (int(x_trans[start]), int(y_trans[start])), (int(x_trans[end]), int(y_trans[end])), COLORS[i], 4)

    return image
        

# --- Main Script ---
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect test images
test_images = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
print(f"Found {len(test_images)} test images.")

# Load the model
print("Loading model...")
my_model = load_model(MODEL_PATH)
print("Model loaded.")

# Define keypoints and associated connections
KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
KEYPOINT_CONNECTIONS = get_limbs_from_keypoints(KEYPOINTS)

# Define color palette for keypoints
COLORS = [
    (0, 255, 255),  # Right Eye to Nose
    (255, 0, 0),    # Right Eye to Right Ear
    (0, 255, 0),    # Left Eye to Nose
    (255, 0, 255),  # Left Eye to Left Ear
    (0, 0, 255),    # Right Shoulder to Right Elbow
    (255, 255, 0),  # Right Elbow to Right Wrist
    (255, 255, 255),# Left Shoulder to Left Elbow
    (0, 128, 255),  # Left Elbow to Left Wrist
    (0, 76, 153),   # Right Hip to Right Knee
    (255, 51, 255), # Right Knee to Right Ankle
    (204, 102, 0),  # Left Hip to Left Knee
    (0, 204, 102),  # Left Knee to Left Ankle
    (204, 0, 204),  # Right Shoulder to Left Shoulder
    (0, 255, 255),  # Right Hip to Left Hip
    (0, 255, 0),    # Right Shoulder to Right Hip
    (0, 204, 102)   # Left Shoulder to Left Hip
]

# Process each test image
for i, image_file in enumerate(test_images):
    image_path = os.path.join(TEST_DIR, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image {image_file}")
        continue  # Skip to the next image if there's an error

    # Resize and normalize the image for the model
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image_resized = image_resized / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    # Predict keypoints
    coordinates = my_model.predict(image_resized)

    print(f"Predicted coordinates for image {i+1}/{len(test_images)}: {coordinates}")

    # Draw the skeleton on the original image
    image = draw_skeleton(image, coordinates)

    # Save the image with drawn skeleton
    output_path = os.path.join(OUTPUT_DIR, image_file)
    if cv2.imwrite(output_path, image):
        print(f"Saved result to {output_path}")
    else:
        print(f"Error saving image to {output_path}")

print("Processing complete.")
