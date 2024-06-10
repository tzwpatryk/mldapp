from flask import Flask, request, jsonify
import torch
import io
from torchvision import transforms
from PIL import Image
from web3 import Web3
import json
import http.client
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS
import requests
from solcx import compile_source, install_solc

app = Flask(__name__)
CORS(app)

# Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
w3.eth.default_account = w3.eth.accounts[0]

# Install Solidity compiler
install_solc('0.8.0')

# Solidity contract source code
contract_source_code = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelStorage {
    mapping(uint256 => string) private modelHashes;
    mapping(uint256 => string) private modelClasses;
    uint256 public modelCount;

    function setModelHash(string memory _hash, string memory _classes) public {
        modelHashes[modelCount] = _hash;
        modelClasses[modelCount] = _classes;
        modelCount++;
    }

    function getModelHash(uint256 index) public view returns (string memory) {
        require(index < modelCount, "Model does not exist");
        return modelHashes[index];
    }

    function getModelClasses(uint256 index) public view returns (string memory) {
        require(index < modelCount, "Model does not exist");
        return modelClasses[index];
    }

    function getModelCount() public view returns (uint256) {
        return modelCount;
    }
}
'''

# Compile the contract
compiled_sol = compile_source(contract_source_code, solc_version='0.8.0')
contract_interface = compiled_sol['<stdin>:ModelStorage']

# Deploy the contract
ModelStorage = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
tx_hash = ModelStorage.constructor().transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

# Get the contract address
contract_address = tx_receipt.contractAddress

# Save ABI and contract address to files
with open('ModelStorage_abi.json', 'w') as abi_file:
    json.dump(contract_interface['abi'], abi_file)

with open('ModelStorage_address.txt', 'w') as address_file:
    address_file.write(contract_address)

# Create contract instance
model_storage = w3.eth.contract(address=contract_address, abi=contract_interface['abi'])

def load_model_from_ipfs(ipfs_hash, base_model, n_classes):
    try:
        conn = http.client.HTTPConnection('127.0.0.1', 8080)
        conn.request('GET', f'/ipfs/{ipfs_hash}')
        response = conn.getresponse()
        if response.status == 200:
            model_bytes = response.read()
            model_buffer = io.BytesIO(model_bytes)

            if base_model == 'mobilenetv2':
                model = models.mobilenet_v2()
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, n_classes)
            elif base_model == 'resnet18':
                model = models.resnet18()
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, n_classes)

            model.load_state_dict(torch.load(model_buffer, map_location="cpu"))
            model.eval()
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transformation(image).unsqueeze(0)
    return image

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400

    model_file = request.files['model']
    num_classes = int(request.form['num_classes'])
    class_names = json.loads(request.form['class_names'])

    # Upload model to IPFS
    response = requests.post('http://127.0.0.1:5001/api/v0/add', files={'file': model_file})
    if response.status_code == 200:
        ipfs_hash = response.json()['Hash']
        
        # Save model hash and classes to the smart contract
        model_storage.functions.setModelHash(ipfs_hash, json.dumps(class_names)).transact()
        
        return jsonify({'ipfs_hash': ipfs_hash}), 200
    else:
        return jsonify({'error': 'Failed to upload model to IPFS'}), 500

@app.route('/get_models', methods=['GET'])
def get_models():
    model_count = model_storage.functions.getModelCount().call()
    models = [{'id': i} for i in range(model_count)]
    return jsonify(models), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model_id' not in request.form or 'base_model' not in request.form:
        return jsonify({'error': 'File, model ID, or base model not provided'}), 400

    model_id = int(request.form['model_id'])
    base_model = request.form['base_model']

    model_ipfs_hash = model_storage.functions.getModelHash(model_id).call()
    model_classes = json.loads(model_storage.functions.getModelClasses(model_id).call())    
    
    if base_model == 'mobilenetv2':
        model = models.mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(model_classes))
    elif base_model == 'resnet18':
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(model_classes))
    else:
        return jsonify({'error': 'Invalid base model type'}), 400

    n_classes = len(model_classes)

    model = load_model_from_ipfs(model_ipfs_hash, base_model, n_classes)
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    input_tensor = process_image(image)
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    class_idx = probabilities.argmax().item()
    class_name = model_classes[str(class_idx)]

    return jsonify({'class_id': class_name}), 200

if __name__ == '__main__':
    app.run(port=5050, debug=True)
