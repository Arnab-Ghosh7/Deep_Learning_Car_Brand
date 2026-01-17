# Deep Learning Car Brand Classification

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.2%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-1.1.2-green?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)


<div align="center">
  <img src="https://www.carlogos.org/car-logos/audi-logo-2016.png" width="100" alt="Audi Logo" style="margin: 0 20px;">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Lamborghini_Logo.svg/1200px-Lamborghini_Logo.svg.png" width="90" alt="Lamborghini Logo" style="margin: 0 20px;">
  <img src="https://www.carlogos.org/car-logos/mercedes-benz-logo.png" width="100" alt="Mercedes Logo" style="margin: 0 20px;">
</div>


<h3 align="center">Advanced Car Brand Recognition System</h3>

---

## 📌 Project Overview
In the rapidly evolving landscape of **Intelligent Transportation Systems (ITS)** and **Automated Surveillance**, the ability to accurately identify vehicle attributes is paramount. This project presents a robust **Deep Learning Car Brand Classification System** designed to bridge the gap between raw image data and actionable insights.

Utilizing the state-of-the-art **ResNet50 (Residual Network)** architecture, this application tackles the challenging task of fine-grained image classification. Unlike traditional shallow networks, ResNet50 employs "skip connections" to allow for training much deeper networks without the vanishing gradient problem, making it exceptionally powerful for extracting complex features from vehicle images.

We have integrated this powerful manufacturing-grade backend with a lightweight, responsive **Flask** web application. This allows users—ranging from automotive enthusiasts to parking management systems—to simply upload an image and receive an instant, high-confidence prediction of the car manufacterer.



## ✨ Features

*   **Deep Transfer Learning**: Implements the 50-layer Residual Network (ResNet50) for superior feature extraction.
*   **High-Performance Accuracy**: The model boasts a training accuracy of ~98% and robust validation performance.
*   **Interactive Web Interface**: A clean, Bootstrap-powered UI allowing drag-and-drop image uploads.
*   **Instant Inference**: Optimized prediction pipeline providing results in milliseconds.
*   **Scalable Architecture**: Codebase structured for easy addition of new car classes or model upgrades.

## 🛠️ Tech Stack & Tools

| Category | Technologies |
| :--- | :--- |
| **Deep Learning** | Python, TensorFlow, Keras, ResNet50 |
| **Backend** | Flask (Python Web Framework) |
| **Image Processing** | OpenCV, Pillow (PIL), NumPy |
| **Frontend** | HTML5, CSS3, Bootstrap 4 |
| **Environment** | Jupyter Notebook (for training), Anaconda |

## 📂 Project Structure

```bash
Deep-Learning-Car-Brand/
├── Datasets/                  # Source images for training and validation
│   ├── train/                 # Training set organized by brand folders
│   └── test/                  # Testing/Validation set
├── static/                    # Frontend assets
│   ├── css/                   # Custom stylesheets
│   └── js/                    # JavaScript files
├── templates/                 # Application views
│   └── index.html             # Main upload interface
├── app.py                     # Flask entry point & inference logic
├── requirements.txt           # Dependency lock file
├── README.md                  # Documentation
└── model_resnet50.h5          # Serialized ResNet50 model weights
```

## 🚀 Installation & Setup

Get the application running on your local machine in minutes.

### Prerequisites
*   Python 3.7+ installed.
*   A package manager like `pip`.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Arnab-Ghosh7/Deep_Learning_Car_Brand
    cd Deep-Learning-Car-Brand
    ```

2.  **Create a Virtual Environment** (Best Practice)
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install TensorFlow, Keras, Flask, and other necessary libraries.*

4.  **Download/Train Model**
    *   If `model_resnet50.h5` is not present, you can run the training notebook `Transfer Learning Resnet 50.ipynb` to generate it.

## 🎮 Usage Guide

1.  **Start the Server**
    ```bash
    python app.py
    ```
    You should see output indicating the server is running on `http://127.0.0.1:5000`.

2.  **Launch the App**
    Open your web browser and go to `http://127.0.0.1:5000`.

3.  **Classify Vehicles**
    *   Click the **Upload** area.
    *   Select an image of an **Audi**, **Lamborghini**, or **Mercedes**.
    *   Hit **Predict**. The system will analyze the image and display the brand name with confidence.

## 🧠 Model Architecture Details

The **ResNet50** model used here is pre-trained on **ImageNet**, a dataset of over 14 million images. We removed the top (classification) layer and replaced it with our custom layers:
1.  **Input Layer**: Accepts images resized to `224x224` pixels.
2.  **ResNet50 Base**: Feature extractor with frozen weights (initially) to retain learned patterns.
3.  **Global Average Pooling**: Reduces spatial dimensions.
4.  **Dense Layer**: Fully connected layer for the final classification into 3 categories.
5.  **Softmax Activation**: Outputs probability distribution across the classes.

## 🤝 Contributing

We welcome contributions!
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request


## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
<p align="center">
  <small>Deep Learning Car Brand Classification Project</small>
</p>
