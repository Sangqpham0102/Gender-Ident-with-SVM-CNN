### Gender Classification using OpenCV and HOG

This Python code utilizes OpenCV, HOG (Histogram of Oriented Gradients), and a pre-trained machine learning model to perform gender classification based on images or a webcam stream. It demonstrates the detection of faces in images or through a webcam feed, extracting features using HOG, and predicting the gender of detected faces.

#### Requirements
- Python
- OpenCV
- Numpy
- Matplotlib
- scikit-image
- joblib

#### Pre-trained Models
Before running the code, download the following pre-trained models and place them in the specified directories:
- [Model](/gender-classification-dataset/Model/model2.sav)
- [Haar Cascade](/gender-classification-dataset/haarcascade_frontalface_default.xml)

#### Usage
The `checking` function checks the gender of the person in an image using a pre-trained model. Additionally, the `webcam` function enables real-time gender detection through the webcam.

Example usage:
```python
# Example to check gender in an image
checking('path/to/image.jpg')

# Example to use webcam for real-time gender detection
webcam()
