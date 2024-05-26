import requests

def upload_to_ipfs(filepath):
    with open(filepath, 'rb') as file:
        files = {'file': (filepath, file, 'application/octet-stream')}
        response = requests.post('http://127.0.0.1:5001/api/v0/add', files=files)
        if response.status_code == 200:
            return response.json()['Hash']
        else:
            raise Exception(f"Failed to upload file to IPFS: {response.text}")

if __name__ == '__main__':
    model_path = 'model.pt'
    ipfs_hash = upload_to_ipfs(model_path)
    print(f'Model IPFS Hash: {ipfs_hash}')
