import os
from flask import Flask, request, render_template, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import io
import base64

# Import the model from the parent directory
import sys
sys.path.append('..')
from model import DenoisingAutoencoderMini, DenoisingAutoencoder

print("HGeqqqqt")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load('../model_resources/1745477403/1745477403_best_model.pth', map_location=device))
model.eval()

def denoise_image(image):
    # Store original size
    original_size = image.size
    
    # Resize image to a fixed size (e.g., 256x256) for processing
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    image_tensor = resize_transform(image).unsqueeze(0).to(device)
    
    # Denoise the image
    with torch.no_grad():
        denoised = model(image_tensor)
    
    # Convert back to PIL Image
    denoised = denoised.squeeze(0).cpu()
    denoised = transforms.ToPILImage()(denoised)
    
    # Resize back to original size
    denoised = denoised.resize(original_size, Image.Resampling.LANCZOS)
    
    return denoised

@app.route('/')
def index():
    print("Accessing index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        try:
            # Read the image
            image = Image.open(file.stream).convert('RGB')
            
            # Denoise the image
            denoised_image = denoise_image(image)
            
            # Save the denoised image
            img_io = io.BytesIO()
            denoised_image.save(img_io, 'JPEG', quality=95)
            img_io.seek(0)
            img_str = base64.b64encode(img_io.getvalue()).decode()
            
            return {'image': f'data:image/jpeg;base64,{img_str}'}
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return {'error': str(e)}, 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True) 