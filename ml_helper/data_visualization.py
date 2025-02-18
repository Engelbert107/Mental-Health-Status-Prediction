import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix



def plot_embeddings(word2vec_reduced, num_words=100, method="tsne", figsize=(12, 8)):
    """
    Plots word embeddings in 2D space using PCA or t-SNE.

    Args:
        word2vec_reduced (dict): Dictionary {word: embedding_vector}.
        num_words (int): Number of words to visualize.
        method (str): Dimensionality reduction method ("pca" or "tsne").
        figsize (tuple): Figure size.
    """
    assert method in ["pca", "tsne"], "Method must be 'pca' or 'tsne'"

    # Select a subset of words to plot
    sampled_words = list(word2vec_reduced.keys())[:num_words]
    sampled_embeddings = np.array([word2vec_reduced[word] for word in sampled_words])

    # Reduce dimensionality
    if method == "pca":
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')

    reduced_embeddings = reducer.fit_transform(sampled_embeddings)

    # Plot
    plt.figure(figsize=figsize)
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], alpha=0.7)

    # Annotate selected words
    for i, word in enumerate(sampled_words):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9, alpha=0.7)

    plt.title(f"Word Embeddings Visualization ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    

def plot_loss(history):
    plt.figure(figsize=(12, 5))
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

def plot_confmat(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()