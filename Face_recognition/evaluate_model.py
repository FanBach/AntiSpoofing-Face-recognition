import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Load the trained face recognition model
def load_embedding_model(model_path):
    return load_model(model_path)

# Generate embeddings for a batch of images
def generate_embeddings(model, image_pairs):
    embeddings = []
    for img1_path, img2_path in image_pairs:
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        embed1 = model.predict(np.expand_dims(img1, axis=0))[0]
        embed2 = model.predict(np.expand_dims(img2, axis=0))[0]
        embeddings.append((embed1, embed2))
    return embeddings

# Preprocess images (resize and normalize)
def preprocess_image(image_path, img_size=(160, 160)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img = tf.keras.utils.img_to_array(img)
    img /= 255.0  # Normalize to [0, 1]
    return img

# Compute similarity scores
def compute_similarity(embedding1, embedding2, metric="cosine"):
    if metric == "cosine":
        return cosine_similarity([embedding1], [embedding2])[0][0]
    elif metric == "euclidean":
        return -np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

# Evaluate model with ROC and AUC
def evaluate_model(embeddings, labels, metric="cosine"):
    scores = []
    true_labels = []
    
    for (embed1, embed2), label in zip(embeddings, labels):
        score = compute_similarity(embed1, embed2, metric=metric)
        scores.append(score)
        true_labels.append(label)
    
    # Compute ROC and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

# Plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, metric, save_path="roc_curve.png"):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({metric} Similarity)')
    plt.legend(loc='lower right')
    plt.savefig(save_path)  # Save the plot
    print(f"ROC curve saved as {save_path}")

# Main function
if __name__ == "__main__":
    # Path to the trained model
    model_path = "face_recognition_model.h5"  # Update this with your trained model path

    # Path to test image pairs and labels
    test_pairs = [("data/bach/0.jpg", "data/bach/1.jpg"),  # Update these paths
                  ("data/bach/2.jpg", "data/bach/3.jpg")]
    labels = [1, 0]  # 1 = same person, 0 = different person
    
    # Load the embedding model
    model = load_embedding_model(model_path)

    # Generate embeddings for image pairs
    embeddings = generate_embeddings(model, test_pairs)

    # Evaluate model with cosine similarity
    fpr_cosine, tpr_cosine, roc_auc_cosine = evaluate_model(embeddings, labels, metric="cosine")
    print("Cosine Similarity - AUC:", roc_auc_cosine)
    plot_roc_curve(fpr_cosine, tpr_cosine, roc_auc_cosine, metric="Cosine", save_path="roc_curve_cosine.png")

    # Evaluate model with Euclidean distance
    fpr_euclidean, tpr_euclidean, roc_auc_euclidean = evaluate_model(embeddings, labels, metric="euclidean")
    print("Euclidean Distance - AUC:", roc_auc_euclidean)
    plot_roc_curve(fpr_euclidean, tpr_euclidean, roc_auc_euclidean, metric="Euclidean", save_path="roc_curve_euclidean.png")
