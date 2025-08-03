import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
TEST_CSV_PATH = '/kaggle/input/visual-taxonomy/test.csv'
TEST_IMAGE_DIR = '/kaggle/input/visual-taxonomy/test_images'
EFFICIENTNET_IMG_SIZE = (384, 384)  # Image size for EfficientNetV2S
XCEPTION_IMG_SIZE = (299, 299)  # Image size for Xception
INCEPTIONRESNET_IMG_SIZE = (299, 299)  # Image size for InceptionResNetV2
BATCH_SIZE = 128  # Adjust based on your system's capacity
TTA_STEPS = 4  # Total TTA images (Original, 30° right, 30° left, Horizontal Flip)

# Load the test data
test_data = pd.read_csv(TEST_CSV_PATH)
#test_data.rename(columns={'len': 'attr_0'}, inplace=True)  # Rename 'len' to 'attr_0' if needed

# Categories and number of attributes
CATEGORY_ATTRIBUTES = {
    "Men Tshirts": 6,
    "Sarees": 11,
    "Kurtis": 10,
    "Women Tshirts": 9,
    "Women Tops & Tunics": 11
}

# Model and encoder paths map
MODEL_ENCODER_PATHS = { #mention the model and encoder paths downloaded from drive
    
    "Men Tshirts": {
        "efficientnet_model_path": "/kaggle/input/effinetv2sx/keras/default/1/Men Tshirts_final_model (15).keras",
        "xception_model_path": "/kaggle/input/xceptionx/keras/default/1/Men Tshirts_final_model (16).keras",
        "inceptionresnet_model_path": "/kaggle/input/inceptionresnetx/keras/default/1/Men Tshirts_final_model (17).keras",
        "encoder_path": "/kaggle/input/effinetv2sx/keras/default/1/Men Tshirts_encoders (14).pkl"
    },
    "Sarees": {
        "efficientnet_model_path": "/kaggle/input/effinetv2sx/keras/default/1/Sarees_final_model (15).keras",
        "xception_model_path": "/kaggle/input/xceptionx/keras/default/1/Sarees_final_model (16).keras",
        "inceptionresnet_model_path": "/kaggle/input/inceptionresnetx/keras/default/1/Sarees_final_model (17).keras",
        "encoder_path": "/kaggle/input/effinetv2sx/keras/default/1/Sarees_encoders (13).pkl"
    },
    "Kurtis": {
        "efficientnet_model_path": "/kaggle/input/effinetv2sx/keras/default/1/Kurtis_final_model (15).keras",
        "xception_model_path": "/kaggle/input/xceptionx/keras/default/1/Kurtis_final_model (16).keras",
        "inceptionresnet_model_path": "/kaggle/input/inceptionresnetx/keras/default/1/Kurtis_final_model (17).keras",
        "encoder_path": "/kaggle/input/effinetv2sx/keras/default/1/Kurtis_encoders (15).pkl"
    },
    "Women Tshirts": {
        "efficientnet_model_path": "/kaggle/input/effinetv2sx/keras/default/1/Women Tshirts_final_model (14).keras",
        "xception_model_path": "/kaggle/input/xceptionx/keras/default/1/Women Tshirts_final_model (15).keras",
        "inceptionresnet_model_path": "/kaggle/input/inceptionresnetx/keras/default/1/Women Tshirts_final_model (16).keras",
        "encoder_path": "/kaggle/input/effinetv2sx/keras/default/1/Women Tshirts_encoders (11).pkl"
    },
    "Women Tops & Tunics": {
        "efficientnet_model_path": "/kaggle/input/effinetv2sx/keras/default/1/Women Tops _ Tunics_final_model (6).keras",
        "xception_model_path": "/kaggle/input/xceptionx/keras/default/1/Women Tops _ Tunics_final_model (7).keras",
        "inceptionresnet_model_path": "/kaggle/input/inceptionresnetx/keras/default/1/Women Tops_ Tunics_final_model (1).keras",
        "encoder_path": "/kaggle/input/effinetv2sx/keras/default/1/Women Tops_ Tunics_encoders (5).pkl"
    }
    
    # ... other categories with model paths
}

# Helper function to create a centered affine transformation matrix for rotation
def get_centered_affine_transform(degrees, img_size):
    radians = np.deg2rad(degrees)
    cos_val = np.cos(radians)
    sin_val = np.sin(radians)
    cx, cy = img_size[1] / 2, img_size[0] / 2
    
    return [
        cos_val, -sin_val, (1 - cos_val) * cx + sin_val * cy,
        sin_val, cos_val, (1 - cos_val) * cy - sin_val * cx,
        0.0, 0.0
    ]

# Function to apply centered rotation
def rotate_image(image, degrees, img_size):
    transform = get_centered_affine_transform(degrees, img_size)
    return tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=[transform],
        output_shape=img_size,
        interpolation="BILINEAR"
    )[0]

# Function to preprocess and apply TTA
def preprocess_image_with_tta(file_path, img_size):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.adjust_contrast(image, contrast_factor=1.4)
    image = image / 255.0  # Normalize

    # Generate TTA images: original, 30° right, 30° left, and horizontal flip
    images = [
        image,
        rotate_image(image, 19, img_size),
        rotate_image(image, -19, img_size),
        tf.image.flip_left_right(image),
        tf.image.adjust_contrast(image, contrast_factor=1.4)
    ]
    return images

# Load encoders
def load_encoders(encoder_path):
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    return encoders

# Decode predictions
def decode_predictions(predictions, encoders, num_attributes):
    decoded_predictions = []
    for i in range(num_attributes):
        pred = predictions[i]
        predicted_labels = np.argmax(pred, axis=1)
        decoded_labels = encoders[f'attr_{i}'].inverse_transform(predicted_labels)
        decoded_predictions.append(decoded_labels)
    return decoded_predictions

# Prepare submission columns
max_num_attributes = max(CATEGORY_ATTRIBUTES.values())
submission_columns = ['id', 'Category'] + [f'attr_{i}' for i in range(max_num_attributes)]
submission_rows = []

# Process test data category by category
for category, num_attributes in CATEGORY_ATTRIBUTES.items():
    print(f"\nProcessing category: {category}")

    category_test_data = test_data[test_data['Category'] == category].copy()
    if category_test_data.empty:
        print(f"No test data found for category: {category}. Skipping.")
        continue

    paths = MODEL_ENCODER_PATHS[category]
    efficientnet_model_path = paths["efficientnet_model_path"]
    xception_model_path = paths["xception_model_path"]
    inceptionresnet_model_path = paths["inceptionresnet_model_path"]
    encoder_path = paths["encoder_path"]

    if not (os.path.exists(efficientnet_model_path) and os.path.exists(xception_model_path) and os.path.exists(inceptionresnet_model_path) and os.path.exists(encoder_path)):
        print(f"Model or encoder for category '{category}' not found. Skipping.")
        continue

    # Load models and encoders
    efficientnet_model = load_model(efficientnet_model_path, compile=False)
    xception_model = load_model(xception_model_path, compile=False)
    inceptionresnet_model = load_model(inceptionresnet_model_path, compile=False)
    current_encoders = load_encoders(encoder_path)

    # Set weights based on category
    efficientnet_weight = 0.53
    xception_weight = 0.39
    inceptionresnet_weight = 0.08

    # Get the list of image file paths and IDs
    image_ids = category_test_data['id'].astype(str).str.zfill(6).tolist()
    image_paths = [os.path.join(TEST_IMAGE_DIR, f"{image_id}.jpg") for image_id in image_ids]

    # Filter out non-existent images
    valid_indices = [i for i, path in enumerate(image_paths) if os.path.exists(path)]
    image_paths = [image_paths[i] for i in valid_indices]
    image_ids = [category_test_data.iloc[i]['id'] for i in valid_indices]

    # Iterate over batches
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_ids = image_ids[i:i + BATCH_SIZE]

        # Apply TTA for each image in the batch and stack them
        efficientnet_images = np.concatenate([
            np.stack(preprocess_image_with_tta(path, EFFICIENTNET_IMG_SIZE)) for path in batch_paths
        ], axis=0)
        xception_images = np.concatenate([
            np.stack(preprocess_image_with_tta(path, XCEPTION_IMG_SIZE)) for path in batch_paths
        ], axis=0)
        inceptionresnet_images = np.concatenate([
            np.stack(preprocess_image_with_tta(path, INCEPTIONRESNET_IMG_SIZE)) for path in batch_paths
        ], axis=0)

        # Get predictions from all models with TTA
        efficientnet_preds = efficientnet_model.predict(efficientnet_images, verbose=0)
        xception_preds = xception_model.predict(xception_images, verbose=0)
        inceptionresnet_preds = inceptionresnet_model.predict(inceptionresnet_images, verbose=0)

        # Average predictions over TTA steps
        efficientnet_preds = [np.mean(np.split(pred, len(batch_paths)), axis=1) for pred in efficientnet_preds]
        xception_preds = [np.mean(np.split(pred, len(batch_paths)), axis=1) for pred in xception_preds]
        inceptionresnet_preds = [np.mean(np.split(pred, len(batch_paths)), axis=1) for pred in inceptionresnet_preds]

        # Combine predictions with weighted averaging
        combined_preds = [
            (efficientnet_weight * eff_pred + xception_weight * xcep_pred + inceptionresnet_weight * inc_pred)
            for eff_pred, xcep_pred, inc_pred in zip(efficientnet_preds, xception_preds, inceptionresnet_preds)
        ]
        
        decoded_preds = decode_predictions(combined_preds, current_encoders, num_attributes)

        for idx, id_value in enumerate(batch_ids):
            submission_row = {
                'id': int(id_value),
                'Category': category
            }
            for j in range(num_attributes):
                submission_row[f'attr_{j}'] = decoded_preds[j][idx]
            for j in range(num_attributes, max_num_attributes):
                submission_row[f'attr_{j}'] = 'missing_value'
            submission_rows.append(submission_row)

    print(f"Completed predictions for category: {category}")

# Create the submission DataFrame
submission = pd.DataFrame(submission_rows)
submission.fillna('dummy_value', inplace=True)
submission = submission[submission_columns]

# Save the submission file
submission.to_csv('submission.csv', index=False)
submission.rename(columns={'attr_0': 'len'}, inplace=True)
print("\nSubmission file created: submission.csv")
