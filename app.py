from flask import Flask, request, jsonify
import torch
import io
from torchvision import transforms
from PIL import Image
from web3 import Web3
import json
import http.client
import io
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
import subprocess

# Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
w3.eth.default_account = w3.eth.accounts[0]

# subprocess.run(["python", "set_model_hash.py"])
# subprocess.run(["python", "deploy_contract.py"])

# Load ABI and contract address
with open('ModelStorage_abi.json', 'r') as file:
    abi = json.load(file)

with open('ModelStorage_address.txt', 'r') as file:
    contract_address = file.read().strip()

# Create contract instance
model_storage = w3.eth.contract(address=contract_address, abi=abi)

app = Flask(__name__)
CORS(app)

classes = {0: 'Asian Green Bee-Eater',
            1: 'Brown-Headed Barbet',
            2: 'Cattle Egret',
            3: 'Common Kingfisher',
            4: 'Common Myna',
            5: 'Common Rosefinch',
            6: 'Common Tailorbird',
            7: 'Coppersmith Barbet',
            8: 'Forest Wagtail',
            9: 'Gray Wagtail',
            10: 'Hoopoe',
            11: 'House Crow',
            12: 'Indian Grey Hornbill',
            13: 'Indian Peacock',
            14: 'Indian Pitta',
            15: 'Indian Roller',
            16: 'Jungle Babbler',
            17: 'Northern Lapwing',
            18: 'Red-Wattled Lapwing',
            19: 'Ruddy Shelduck',
            20: 'Rufous Treepie',
            21: 'Sarus Crane',
            22: 'White Wagtail',
            23: 'White-Breasted Kingfisher',
            24: 'White-Breasted Waterhen'}

def load_model_from_ipfs(ipfs_hash):
    try:
        conn = http.client.HTTPConnection('127.0.0.1', 8080)
        conn.request('GET', f'/ipfs/{ipfs_hash}')
        response = conn.getresponse()
        print(f"Status Code: {response.status}")
        if response.status == 200:
            model_bytes = response.read()
            model_buffer = io.BytesIO(model_bytes)
            model = models.mobilenet_v2()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 25)
            model.load_state_dict(torch.load(model_buffer, map_location="mps"))
            model.eval()
            print("Model loaded successfully.")
            return model
        else:
            print(f"Failed to fetch model from IPFS: {response.status}")
        conn.close()
    except Exception as e:
        print(f"Error connecting to localhost:8080 - {e}")

def process_image(image):
    transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transformation(image).unsqueeze(0)
    return image_tensor


@app.route('/predict', methods=['POST'])
def predict():
    model_ipfs_hash = model_storage.functions.getModelHash().call()
    model = load_model_from_ipfs(model_ipfs_hash)
    print("mamy model")
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    
    input_tensor = process_image(image)
    
    output = model(input_tensor)
    
    probabilities = F.softmax(output, dim=1)
    class_idx = probabilities.argmax().item()
    class_name = classes[class_idx]
    print(class_name)

    return jsonify({'class_id': class_name}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
