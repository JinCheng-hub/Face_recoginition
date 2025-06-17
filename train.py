import os
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_data(src, imgsz=128):
    data = {'label': [], 'data': []}

    for subdir in os.listdir(src):
        current_path = os.path.join(src, subdir)
        for file in os.listdir(current_path):
            if file.endswith(('jpg', 'jpeg', 'png')):
                im = cv2.imread(os.path.join(current_path, file))
                im = resize(im, (imgsz, imgsz))
                data['label'].append(subdir)
                data['data'].append(im)
    return np.array(data['data']), np.array(data['label'])

    
def preprocess(images):
    return np.array([
        hog(rgb2gray(img), orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        for img in images
    ])


def train_model(X_train, X_test, y_train, y_test, kernel="linear"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(kernel=kernel, probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = 100 * np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}%")
    return clf, scaler, y_pred


def grid_search_train(X_train, y_train):
    param_grid = [
        {
            'estimator': [SVC(probability=True, random_state=42)],
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__gamma': [1, 0.1, 0.01, 0.001],
            'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        },
        {
            'estimator': [SGDClassifier('log_loss', random_state=42)],
            'estimator__penalty': ['l2', 'l1', 'elasticnet'],
            'estimator__alpha': [0.0001, 0.001, 0.01],
        }
    ]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', SVC())
    ])

    grid = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print("Best Parameters:", grid.best_params_)
    print("Best Cross-Validation Score:", grid.best_score_)
    return grid.best_estimator_


def plot_confusion_matrix(cmx, labels, path):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    titles = ['Raw Counts', 'Normalized (%)', 'Normalized w/o Diagonal']
    matrices = [cmx, cmx_norm, cmx_zero_diag]

    for i in range(3):
        im = ax[i].imshow(matrices[i], cmap='Blues')
        ax[i].set_title(titles[i])
        ax[i].set_xticks(np.arange(len(labels)))
        ax[i].set_yticks(np.arange(len(labels)))
        ax[i].set_xticklabels(labels, rotation=45)
        ax[i].set_yticklabels(labels)
        for x in range(len(labels)):
            for y in range(len(labels)):
                value = matrices[i][x, y]
                ax[i].text(y, x, f"{value:.1f}" if i > 0 else int(value),
                           ha='center', va='center', color='black')

        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    fig.tight_layout()
    plt.savefig(os.path.join(path, "Confusion_Matrix.png"))
    plt.close()


def save_model_and_report(clf, scaler, label_encoder, y_test, y_pred, dataset, X_test_original, exp_path):
    # Save model
    with open(os.path.join(exp_path, "model.pkl"), 'wb') as f:
        pickle.dump({"classifier": clf, "scaler": scaler, "labels": label_encoder}, f)
    
    # Save report
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, label_encoder.classes_, exp_path)
    with open(os.path.join(exp_path, "report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\nDataset: " + dataset)
        f.write("\nTraining Parameters:\n")
        f.write(str(clf.get_params()))
    
    # Save misclassified
    os.makedirs(os.path.join(exp_path, "misclassified"), exist_ok=True)
    for idx in np.where(y_test != y_pred)[0]:
        img = rgb2gray(X_test_original[idx])
        plt.imshow(img, cmap='gray')
        true = label_encoder.inverse_transform([y_test[idx]])[0]
        pred = label_encoder.inverse_transform([y_pred[idx]])[0]
        plt.title(f"True: {true}, Pred: {pred}")
        plt.axis('off')
        plt.savefig(os.path.join(exp_path, "misclassified", f"{idx}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="faces", help="Dataset folder")
    parser.add_argument("--size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--kernel", default="linear", choices=["linear", "poly", "rbf", "sigmoid"])
    parser.add_argument("--grid_search", action="store_true", help="Use grid search")
    args = parser.parse_args()

    # Load and preprocess data
    X, y = load_data(args.dataset)
    X_features = preprocess(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test, _, X_test_orig = train_test_split(
        X_features, y_encoded, X, test_size=args.size, shuffle=True, stratify=y_encoded, random_state=42)

    # Train model
    if args.grid_search:
        clf = grid_search_train(X_train, y_train)
        y_pred = clf.predict(X_test)
        scaler = None
    else:
        clf, scaler, y_pred = train_model(X_train, X_test, y_train, y_test, args.kernel)

    # Save
    os.makedirs("train", exist_ok=True)
    exp_id = str(max([int(d) for d in os.listdir("train") if d.isdigit()] + [0]) + 1)
    exp_path = os.path.join("train", exp_id)
    os.makedirs(exp_path, exist_ok=True)
    save_model_and_report(clf, scaler, label_encoder, y_test, y_pred, args.dataset, X_test_orig, exp_path)

if __name__ == "__main__":
    main()