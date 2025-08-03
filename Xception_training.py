import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, multiply, Reshape, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging
from tqdm import tqdm
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomBrightness, RandomContrast, RandomCrop
# Suppress TensorFlow warnings for cleaner output
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Define Paths
TRAIN_CSV_PATH = '/kaggle/input/visual-taxonomy/train.csv'
TRAIN_IMAGE_DIR = '/kaggle/input/visual-taxonomy/train_images'

# Image parameters
IMG_SIZE = (299, 299)  # Changed to 299, 299 for Xception compatibility
BATCH_SIZE = 20  # Adjust based on your system's capacity
EPOCHS = 20

# Categories and their respective number of attributes
CATEGORY_ATTRIBUTES = {
    "Men Tshirts": 6,
    "Sarees": 11,
    "Kurtis": 10,
    "Women Tshirts": 9,
    "Women Tops & Tunics": 11
}

# Output directories
MODEL_DIR = './models'
ENCODER_DIR = './encoders'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

# Data Augmentation using Keras preprocessing layers
data_augmentation = tf.keras.Sequential([
    RandomRotation(factor=0.0833),  # Approximately 30 degrees
    RandomFlip("horizontal"),
    #RandomBrightness(factor=0.1)
])

# Load Training Data
train_data = pd.read_csv(TRAIN_CSV_PATH)

# Rename 'len' to 'attr_0' if necessary
if 'len' in train_data.columns:
    train_data.rename(columns={'len': 'attr_0'}, inplace=True)

# Function to Fit LabelEncoders for Each Attribute without treating "missing_value" as a valid class during training
def fit_label_encoders(data, num_attributes):
    encoders = {}
    for i in range(num_attributes):
        attr = f'attr_{i}'
        le = LabelEncoder()
        valid_labels = data[attr].dropna().astype(str).str.strip().str.lower()
        unique_labels = valid_labels[valid_labels != 'missing_value'].unique().tolist()
        le.fit(unique_labels)
        encoders[attr] = le
        print(f"Encoder for {attr} has classes: {le.classes_}")
    return encoders

# Data Generator Class with Masking
class DataGenerator(Sequence):
    def __init__(self, data, img_dir, encoders, img_size, batch_size, num_attributes, shuffle=True, augment=False):
        self.data = data.reset_index(drop=True)
        self.img_dir = img_dir
        self.encoders = encoders
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_attributes = num_attributes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data.iloc[batch_indices]
        images = []
        labels = [[] for _ in range(self.num_attributes)]
        sample_weights = [[] for _ in range(self.num_attributes)]

        for _, row in batch_data.iterrows():
            image_id = str(row['id']).zfill(6)
            image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                try:
                    img = load_img(image_path, target_size=self.img_size)
                    img = img_to_array(img)

                    # Apply augmentations if enabled
                    img = tf.convert_to_tensor(img, dtype=tf.float32)
                    if self.augment:
                        img = data_augmentation(img)
                    img = tf.image.adjust_contrast(img, contrast_factor=1.4).numpy()

                    # Normalize the image
                    img = img / 255.0
                    images.append(img)

                    for i in range(self.num_attributes):
                        attr = f'attr_{i}'
                        label = row[attr]
                        if pd.isnull(label) or label == '' or label == 'missing_value':
                            label_encoded = -1
                            sample_weights[i].append(0.0)  # Zero weight for missing labels
                        else:
                            label = str(label).strip().lower()
                            if label in self.encoders[attr].classes_:
                                label_encoded = self.encoders[attr].transform([label])[0]
                                sample_weights[i].append(1.0)  # Full weight for valid labels
                            else:
                                label_encoded = -1
                                sample_weights[i].append(0.0)  # Zero weight for invalid labels

                        if label_encoded >= 0:
                            label_one_hot = to_categorical(label_encoded, num_classes=len(self.encoders[attr].classes_))
                        else:
                            label_one_hot = np.zeros(len(self.encoders[attr].classes_), dtype=np.float32)

                        labels[i].append(label_one_hot)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}. Skipping.")
                    continue
            else:
                print(f"Image not found: {image_path}. Skipping.")
                continue

        images = np.array(images, dtype=np.float32)
        labels = [np.array(label_list, dtype=np.float32) for label_list in labels]
        sample_weights = [np.array(weight_list, dtype=np.float32) for weight_list in sample_weights]

        return images, tuple(labels), tuple(sample_weights)

# Define SE Block
def se_block(input_tensor, reduction_ratio=16):
    channel_axis = -1
    filters = tf.keras.backend.int_shape(input_tensor)[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=False, dtype='float32')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, dtype='float32')(se)
    
    x = multiply([input_tensor, se])
    return x

# Build Model Function with SE Blocks
def build_model_with_se(img_size, num_attributes, encoders):
    input_layer = Input(shape=(*img_size, 3))
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)
    
    x = base_model.output
    x = se_block(x, reduction_ratio=16)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    outputs = []
    for i in range(num_attributes):
        attr = f'attr_{i}'
        num_classes = len(encoders[attr].classes_)
        output = Dense(num_classes, activation='softmax', name=attr)(x)
        outputs.append(output)
    
    model = Model(inputs=input_layer, outputs=outputs)
    return model, base_model

# AdamW Optimizer Implementation without Addons
class AdamW(tf.keras.optimizers.Adam):
    def __init__(self, weight_decay=1e-4, *args, **kwargs):
        super(AdamW, self).__init__(*args, **kwargs)
        self.weight_decay = weight_decay

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if self.weight_decay > 0:
            var.assign_sub(self.weight_decay * var, use_locking=self._use_locking)
        return super(AdamW, self)._resource_apply_dense(grad, var, apply_state)

# Compile Model Function
def compile_model(model, num_attributes, weight_decay=2e-4):
    loss_functions = [tf.keras.losses.CategoricalCrossentropy() for _ in range(num_attributes)]
    metrics = ['accuracy'] * num_attributes

    model.compile(
        optimizer=AdamW(learning_rate=1e-4, weight_decay=weight_decay),
        loss=loss_functions,
        metrics=metrics
    )

    return model

# Callbacks Function
def get_callbacks(category):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"{category}_best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    return [early_stopping, reduce_lr, checkpoint]
# Training Loop for Each Category
for category, num_attributes in CATEGORY_ATTRIBUTES.items():
    print(f"\n=== Training for Category: {category} ===")
    tf.keras.backend.clear_session()
    category_data = train_data[train_data['Category'] == category].copy()

    if category_data.empty:
        print(f"No data found for category: {category}. Skipping.")
        continue

    encoders = fit_label_encoders(category_data, num_attributes)

    train_split, val_split = train_test_split(
        category_data,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Training samples: {len(train_split)}, Validation samples: {len(val_split)}")
    train_generator = DataGenerator(
        data=train_split,
        img_dir=TRAIN_IMAGE_DIR,
        encoders=encoders,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_attributes=num_attributes,
        shuffle=True,
        augment=True  # Enable augmentation for training data
    )

    val_generator = DataGenerator(
        data=val_split,
        img_dir=TRAIN_IMAGE_DIR,
        encoders=encoders,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_attributes=num_attributes,
        shuffle=False,
        augment=False  # No augmentation for validation data
    )

    model, base_model = build_model_with_se(IMG_SIZE, num_attributes, encoders)
    model = compile_model(model, num_attributes, weight_decay=2e-4)

    print(model.summary())

    callbacks = get_callbacks(category)

    # Train the model with sample weights
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning: Unfreeze more layers
    for layer in base_model.layers[-130:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=2e-4),
        loss=model.loss,
        metrics=['accuracy'] * num_attributes
    )

    fine_tune_epochs = 15
    total_epochs = EPOCHS + fine_tune_epochs

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks,
        verbose=1
    )

    final_model_path = os.path.join(MODEL_DIR, f"{category}_final_model.keras")
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

    encoder_path = os.path.join(ENCODER_DIR, f"{category}_encoders.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Label encoders saved at: {encoder_path}")
    tf.keras.backend.clear_session()
print("Training completed for all categories.")
