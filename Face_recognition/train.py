import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # <-- This import is missing
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def build_embedding_model(input_shape=(160, 160, 3), num_classes=None):
    """
    Builds the CNN for classification-based training.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes for classification.

    Returns:
        model: Keras model for face recognition.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),  # Embedding layer
        layers.Dense(num_classes, activation='softmax')  # Classification layer
    ])

    return model

def train_model(data_dir, input_shape=(160, 160, 3), batch_size=32, epochs=10):
    """
    Trains the face recognition model using the provided data.

    Args:
        data_dir (str): Path to the dataset directory.
        input_shape (tuple): Shape of the input images.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Returns:
        None
    """
    # Image data generator for training and validation sets
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # Split data into training and validation sets
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Build the model
    model = build_embedding_model(input_shape=input_shape, num_classes=train_gen.num_classes)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    # Save the trained model
    model.save('face_recognition_model.h5')
    print("Model training complete. Model saved as 'face_recognition_model.h5'.")

    # Save the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_gen.classes)
    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(label_encoder, le_file)
    print("Label encoder saved as 'label_encoder.pkl'.")

# Run the training process
if __name__ == "__main__":
    data_dir = "data"  # Path to your dataset
    train_model(data_dir)
