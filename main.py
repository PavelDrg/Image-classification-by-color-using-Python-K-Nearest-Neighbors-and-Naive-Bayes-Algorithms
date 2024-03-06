import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Function to load images and labels from the dataset
def load_dataset(root_folder):
    images = []
    labels = []
    class_mapping = {}  # To map class names to numeric labels

    for class_label, class_name in enumerate(os.listdir(root_folder)):
        class_mapping[class_label] = class_name
        class_folder = os.path.join(root_folder, class_name)

        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = imread(image_path)
            image = resize(image, (64, 64), anti_aliasing=True)  # Adjust image size as needed
            images.append(image.flatten())  # Flatten the image
            labels.append(class_label)
    return np.array(images), np.array(labels), class_mapping


# Function to save results to a CSV file
def save_results(algorithm, y_true, y_pred, class_mapping):
    results_df = pd.DataFrame({'True Label': [class_mapping[label] for label in y_true],
                               'Predicted Label': [class_mapping[label] for label in y_pred]})

    confusion_mat = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, target_names=class_mapping.values(), output_dict=True)

    results_df.to_csv(f'./results/{algorithm}_results.csv', index=False)
    print(f'Results saved for {algorithm} at ./results/{algorithm}_results.csv')

    # Afisam matricea de confuzie cu seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_mapping.values(),
                yticklabels=class_mapping.values())
    plt.title(f'Confusion Matrix - {algorithm}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    with open(f'./results/{algorithm}_metrics.txt', 'w') as metrics_file:
        metrics_file.write(f'Confusion Matrix:\n{confusion_mat}\n\n')
        metrics_file.write(f'Accuracy: {accuracy:.2f}\n\n')
        metrics_file.write(f'Classification Report:\n{classification_rep}')


# Function to save test images with predictions
def save_test_images(algorithm, X_test, y_true, y_pred, class_mapping, probabilities):
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for idx, (true_label, pred_label, prob_dist) in enumerate(zip(y_true, y_pred, probabilities)):
        image = X_test[idx].reshape(64, 64, -1)  # Reshape flattened image
        plt.imshow(image)
        plt.title(f'True Label: {class_mapping[true_label]}\nPredicted Label: {class_mapping[pred_label]}')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_test_image_{idx}.png'))
        plt.close()

        # Plot probability distribution
        plt.bar(class_mapping.values(), prob_dist, color='blue')
        plt.title(f'Probability Distribution - Test Image {idx}')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_probability_distribution_{idx}.png'))
        plt.close()
    print(f'Test images saved for {algorithm} in {results_folder}')


# Load your dataset
dataset_folder = "./dataset"  # Change this to the path of your dataset folder
X, y, class_mapping = load_dataset(dataset_folder)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example 1: k-Nearest Neighbors (k-NN)
# Initialize k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred_knn = knn_classifier.predict(X_test)

# Save results and test images for k-NN
save_results('knn', y_test, y_test_pred_knn, class_mapping)
probabilities_knn = knn_classifier.predict_proba(X_test)
save_test_images('knn', X_test, y_test, y_test_pred_knn, class_mapping, probabilities_knn)

# Example 2: Naive Bayes (Gaussian Naive Bayes)
# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred_nb = nb_classifier.predict(X_test)

# Save results and test images for Naive Bayes
save_results('naive_bayes', y_test, y_test_pred_nb, class_mapping)
probabilities_nb = nb_classifier.predict_proba(X_test)
save_test_images('naive_bayes', X_test, y_test, y_test_pred_nb, class_mapping, probabilities_nb)
