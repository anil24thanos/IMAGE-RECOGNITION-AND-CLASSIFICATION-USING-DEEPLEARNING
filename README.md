# IMAGE-RECOGNITION-AND-CLASSIFICATION-USING-DEEPLEARNING

*NAME* :  ANIL VARIKUPPALA

ENROLLMENT NUMBER : 23STUCHH010311

*ORGANIZATION* : YBI FOUNDATION

DURATION : 45DAYS

*DOMAIN* : DATASCIENCE AND ARTIFICIAL INTELLEGENCE

MENTOR : VAMSINATH J

DESCRIPTION:
# 🧠 Image Classification using Deep Learning (CIFAR-10)

This project demonstrates how to build an **image recognition and classification system** using **Convolutional Neural Networks (CNNs)** and the **CIFAR-10 dataset**, all implemented in Python with **TensorFlow/Keras**. It classifies images into one of 10 classes, such as airplane, car, dog, and ship.

> ✅ No need to upload your own dataset — CIFAR-10 is automatically downloaded via Keras.

---

## 🚀 Features

- 📦 Uses CIFAR-10 dataset (60,000 images, 10 categories)
- 🧠 Builds a CNN from scratch using TensorFlow/Keras
- 📊 Evaluates model performance and plots training accuracy
- 🔍 Makes predictions on unseen test images
- 🖼️ Displays sample images with actual and predicted labels

---

## 🧰 Technologies Used

| Tool         | Purpose                      |
|--------------|-------------------------------|
| Python       | Programming language          |
| TensorFlow/Keras | Deep learning framework     |
| Matplotlib   | Data visualization            |
| NumPy        | Array processing              |
| Google Colab | Development environment (GPU) |

---

## 🗂️ Dataset

- **Name**: CIFAR-10  
- **Size**: 60,000 color images (32x32)  
- **Classes**:  
  `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`

Loaded automatically with:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


How to Run
Open cifar10_classifier.ipynb in Google Colab

Run all cells (training takes ~10-15 mins with GPU)

View accuracy graphs and test predictions

Try changing the CNN architecture to improve results

📊 Results
Typical results after 10 epochs:

📈 Training Accuracy: ~75%

📉 Validation Accuracy: ~70%

Accuracy may vary based on number of epochs and model depth.

🛠 Future Improvements
🔁 Add data augmentation for generalization

📦 Use transfer learning (e.g. MobileNetV2)

🧠 Save and load model using .h5 or .keras format

🎥 Extend to webcam input using OpenCV and Streamlit

🤝 Acknowledgements
CIFAR-10 Dataset from Keras Datasets

TensorFlow and Keras team for open-source deep learning libraries

📜 License
This project is licensed under the MIT License. Feel free to use, modify, and share.

