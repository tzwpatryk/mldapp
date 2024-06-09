import http.client
import io
import torch
from torchvision import models
import torch.nn as nn

with open("hash.txt", "r") as file:
    ipfs_hash = file.read().strip()

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
        print(model)
    else:
        print(f"Failed to fetch model from IPFS: {response.status}")
    conn.close()
except Exception as e:
    print(f"Error connecting to localhost:8080 - {e}")
