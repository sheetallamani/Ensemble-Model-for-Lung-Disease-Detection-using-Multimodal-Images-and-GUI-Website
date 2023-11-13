# Ensemble Model for Lung Disease Detection using Multimodal-Images and GUI-Website
![GitHub Logo](https://raw.githubusercontent.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/main/Images/1st%20page.png)
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/fine.png?raw=true)

## Description
 
This project implements a web application for detecting various chest conditions, including pneumonia in X-ray images (indicated as "Pneumonia") and different types of lung cancers, such as Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, and COVID-19 in CT scans. The application utilizes two deep learning models: VGG16 for X-ray images and ResNet-50 for CT scans. These models are served through a Flask web server, allowing users to upload chest images and receive predictions for a range of respiratory conditions.

## Adenocarcinoma
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/andr.png?raw=true)
## Large cell carcinoma
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/lar.png?raw=true)
## Squamous cell carcinoma
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/sm.png?raw=true)
## Pneumonia
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/pno.png?raw=true)

## Accuracy  Table
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/ll.png?raw=true)

## Resnet-50 Architectures
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/resnet.png?raw=true)
## VGG-16 Architectures
![GitHub Logo](https://github.com/sheetallamani/Ensemble-Model-for-Lung-Disease-Detection-using-Multimodal-Images-and-GUI-Website/blob/main/Images/vgg.png?raw=true)
## Features

- Image upload functionality
- Prediction using two different models
- User-friendly web interface

## Technologies Used

- Python
- Flask
- TensorFlow
- Keras
- Bootstrap (for the frontend)
- jQuery (for handling asynchronous requests)

## Models

1. **Model 1**: ResNet50-based model for detecting pneumonia in CT scans.
2. **Model 2**: VGG16-based model for detecting pneumonia in X-ray images.

## Getting Started

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask app using `python app.py`.
4. Open the web application in your browser.

## Usage

1. Access the home page and navigate to the 'Upload Report' section.
2. Choose an image file (.png, .jpg, .jpeg) for prediction.
3. Click the 'Detect' button to get predictions from both models.
4. View the results displayed on the web page.


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Include credits or acknowledgments if needed.


