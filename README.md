### Gender Classification Model
This repository contains the source code for a gender classification model that predicts gender based on facial images. The model utilizes computer vision techniques and machine learning algorithms to categorize individuals into male or female classes.

#### Model Details
- The model employs a combination of feature extraction using Histogram of Oriented Gradients (HOG) and a machine learning classifier.
- Note that due to missing files (`model2.sav` and `haarcascade_frontalface_default.xml`), the model cannot be executed directly.
- To retrain the model and generate the missing files, run the code in `Source_code_svm.ipynb`. This notebook assists in retraining the model to create `model2.sav` and `haarcascade_frontalface_default.xml`.
- The training data used to create the missing files can be found in the [Kaggle Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset).

#### How to Retrain the Model
To retrain the model and create the missing files, follow these steps:

1. **Open `Source_code_svm.ipynb`:** Open this notebook using Jupyter Notebook or Google Colab.
2. **Run the Code Cells:** Execute the code cells in the notebook to retrain the model. Ensure you have all the necessary resources like training data and required libraries before running the code.
3. **Save the New Model:** After the training is complete, make sure to save the new model with the name `model2.sav`.

Once done, you can use the newly generated files to perform gender predictions or classifications in your applications.

### Gender Classification using OpenCV and HOG
The provided `Demo.py` script enables gender classification on images or through a webcam using the retrained model files (`model2.sav` and `haarcascade_frontalface_default.xml`). This script involves detecting faces in images or live through a webcam feed, extracting features using HOG, and making gender predictions based on the detected faces. To use this script, ensure you have the following libraries installed:

- `OpenCV`
- `Numpy`
- `Matplotlib`
- `scikit-image`
- `joblib`
- `tkinter` (for GUI components if using the provided script)

Ensure these libraries are installed in your Python environment before executing the provided Python script for gender classification.
