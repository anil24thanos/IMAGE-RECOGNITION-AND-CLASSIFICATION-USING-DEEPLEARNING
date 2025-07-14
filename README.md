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



