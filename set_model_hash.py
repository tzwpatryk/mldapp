from web3 import Web3
import json

# Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
w3.eth.default_account = w3.eth.accounts[0]

# Load ABI and contract address
with open('ModelStorage_abi.json', 'r') as file:
    abi = json.load(file)

with open('ModelStorage_address.txt', 'r') as file:
    contract_address = file.read().strip()

# Create contract instance
model_storage = w3.eth.contract(address=contract_address, abi=abi)

# Set the IPFS hash of the model
with open("hash.txt", "r") as file:
    ipfs_hash = file.read().strip()
model_ipfs_hash = ipfs_hash
tx_hash = model_storage.functions.setModelHash(model_ipfs_hash).transact()
w3.eth.wait_for_transaction_receipt(tx_hash)
print(f'Model IPFS Hash set in the contract: {model_ipfs_hash}')
