from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
import io
import base64
from transformers import pipeline
import torch
import torch.nn.functional as F
import ollama
import yaml
from io import BytesIO
import logging


# Initialize Flask app
app = Flask(__name__)

# Load the depth estimation model
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json['image']  # Base64 encoded frame from the video
    prompt_path = request.json.get('prompt_path', 'prompt/prompt.yml')  # Optional: default path if not provided
    frame = decode_image(data)
    depth_map = apply_depth_estimation(pipe, frame)
    encoded_depth_map = encode_image(depth_map)
    response_message = interact_with_llm(frame, depth_map, prompt_path)
    return jsonify({'depth_map': encoded_depth_map, 'message': response_message})

def decode_image(data):
    header, encoded = data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(binary_data))
    return np.array(img)

def encode_image_to_bytes(image):
    # Check if the input is a NumPy array and convert to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Now we are sure 'image' is a PIL Image, proceed with saving
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)  # Move the cursor to the beginning of the stream
    return image_bytes.getvalue()

def apply_depth_estimation(pipe, img):
    pil_img = Image.fromarray(img)
    original_width, original_height = pil_img.size
    depth = pipe(pil_img)["depth"]
    depth_tensor = torch.from_numpy(np.array(depth)).unsqueeze(0).unsqueeze(0).float()
    depth_resized = F.interpolate(depth_tensor, size=(original_height, original_width), mode='bilinear', align_corners=False)[0, 0]
    depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
    depth_normalized_np = depth_normalized.byte().cpu().numpy()
    colored_depth = cv2.applyColorMap(depth_normalized_np, cv2.COLORMAP_INFERNO)
    colored_depth_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_depth_rgb)

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded

def get_prompt(prompt_path:str):
    with open(prompt_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg['generator_prompt']

def encode_image_to_bytes(image):
    # Check if the input is a NumPy array and convert to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Now we are sure 'image' is a PIL Image, proceed with saving
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)  # Move the cursor to the beginning of the stream
    return image_bytes.getvalue()

def calculate_scene_change(depth_map_a, depth_map_b, threshold=1000):
    # Ensure both depth maps are numpy arrays and compute the absolute difference
    if not (isinstance(depth_map_a, np.ndarray) and isinstance(depth_map_b, np.ndarray)):
        raise ValueError("Both depth_map_a and depth_map_b must be numpy arrays.")

    # Compute the absolute difference and sum it
    difference = np.abs(depth_map_a - depth_map_b)
    total_difference = np.sum(difference)

    # Check if the total difference exceeds the threshold
    flag = total_difference > threshold

    return flag

def load_zoedepth_model(model_name='ZoeD_N', source='local'):
    model = torch.hub.load('isl-org/ZoeDepth', model_name, source=source, pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    return model, DEVICE

def predict_depth(image_path, model, DEVICE):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    tensor_image=image
    # Predict depth
    with torch.no_grad():  # Inference only, no gradients needed
        depth_tensor = model.infer(tensor_image)
    
    # Convert depth tensor to NumPy array for easier manipulation/use
    depth_numpy = depth_tensor.squeeze().cpu().numpy()
    return depth_numpy


def interact_with_llm(image, depth_map, prompt_path:str):
    # Check if the image or depth_map is None or has no elements
    if image is None or depth_map is None or image.size == 0 or depth_map.size == 0:
        logging.error("One of the images is None or empty")
        return "Error: Image or Depth Map is missing or empty."

    prompt_text = get_prompt(prompt_path)
    image_bytes = encode_image_to_bytes(image)
    depth_map_bytes = encode_image_to_bytes(depth_map)

    if image_bytes is None or depth_map_bytes is None:
        return "Failed to encode images."

    response = ollama.chat(model='llava:7b-v1.5-q2_K', messages=[
        {
            'role': 'user',
            'content': prompt_text,
            'images': [image_bytes, depth_map_bytes]
        }
    ])
    return response['message']['content']

if __name__ == '__main__':
    app.run(debug=True, port=5001)
