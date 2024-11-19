# **Image Classification Using Support Vector Machine (SVM)**

This project implements an **SVM-based image classification model** to classify images of cats and dogs. The model is trained and tested using a dataset of labeled images, enabling it to distinguish between the two categories.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Dependencies](#dependencies)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## **Project Overview**

Image classification is a crucial problem in machine learning and computer vision. This project uses a **Support Vector Machine (SVM)**, a supervised machine learning algorithm, to classify images of cats and dogs. The model is built using features extracted from images to train a binary classifier.

---

## **Dataset**

The dataset used in this project contains labeled images of cats and dogs, sourced from the **Kaggle Cats vs. Dogs dataset**. Each image is resized and preprocessed before being fed into the SVM model.

- **Dataset Source**: [Kaggle Cats vs. Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Categories**:
  - **Cats**: Images labeled as "cat".
  - **Dogs**: Images labeled as "dog".

### Example Data:

| Image         | Label |
|---------------|-------|
| cat_image.jpg | Cat   |
| dog_image.jpg | Dog   |

---

## **Features**

1. **Preprocessing**: Resize, grayscale conversion, and normalization of images.
2. **Feature Extraction**: Extract pixel values or histogram of oriented gradients (HOG) as features.
3. **Model Training**: Use SVM with a radial basis function (RBF) kernel for classification.
4. **Evaluation**: Compute metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

## **Dependencies**

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `opencv-python`

Install all dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python
```

---

## **Project Workflow**

1. **Load Dataset**:
   - Download and extract the Cats vs. Dogs dataset.
   - Split the dataset into training and testing sets.
2. **Preprocessing**:
   - Resize all images to a fixed size (e.g., 64x64).
   - Convert images to grayscale.
   - Normalize pixel values to the range [0, 1].
3. **Feature Extraction**:
   - Use pixel intensities or HOG features for SVM input.
4. **Model Training**:
   - Train an SVM classifier using the RBF kernel.
5. **Evaluation**:
   - Test the model on the test dataset.
   - Compute classification metrics (accuracy, precision, recall, F1-score).
6. **Visualization**:
   - Visualize sample predictions using matplotlib.

---

## **Results**

- **Accuracy**: The SVM classifier achieved an accuracy of ~90% on the test dataset.
- **Misclassified Examples**: Some challenging examples (e.g., blurry images or mixed features) were misclassified.
- **Confusion Matrix**:
  
|              | Predicted Cat | Predicted Dog |
|--------------|---------------|---------------|
| Actual Cat   | 48            | 2             |
| Actual Dog   | 3             | 47            |

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-dog-classification-svm.git
   cd cat-dog-classification-svm
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   - Download the dataset from Kaggle and extract it into the `data/` folder.

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook svm_cat_dog_classification.ipynb
   ```

5. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

---

## **Screenshots**

### 1. **Sample Images from the Dataset**
![Dataset Sample](https://via.placeholder.com/800x400?text=Sample+Images+from+Dataset)

### 2. **Confusion Matrix**
![Confusion Matrix](https://via.placeholder.com/800x400?text=Confusion+Matrix)

### 3. **Predictions Visualization**
![Predictions Visualization](https://via.placeholder.com/800x400?text=Predictions+Visualization)

---

## **Future Enhancements**

- Include data augmentation to improve generalization.
- Experiment with deep learning models like CNNs for better accuracy.
- Create a GUI to upload images and view predictions in real time.

---

## **License**

This project is licensed under the [MIT License](LICENSE).
