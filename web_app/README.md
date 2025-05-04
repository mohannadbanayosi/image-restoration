# Image Denoising Web App

A lightweight web application that allows users to upload noisy images and get them denoised using a trained deep learning model.

## Features

- Drag and drop interface for image upload
- Real-time image denoising
- Side-by-side comparison of original and denoised images
- Modern and responsive UI

## Prerequisites

- Python 3.8 or higher
- Trained denoising model (should be in the parent directory's `model_resources` folder)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you have activated your virtual environment
2. Run the Flask application:
```bash
python app.py
```
3. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Click on the upload area or drag and drop an image file
2. Wait for the denoising process to complete
3. View the original and denoised images side by side

## Notes

- The application supports JPG, PNG, and JPEG image formats
- Maximum file size is limited to 16MB
- The denoising process may take a few seconds depending on the image size and your hardware 