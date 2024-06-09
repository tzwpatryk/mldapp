from web3 import Web3
from solcx import compile_source, install_solc
import json

# Connect to local Ethereum node provided by Ganache
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
w3.eth.default_account = w3.eth.accounts[0]

# Install Solidity compiler
install_solc('0.8.0')

# Read the Solidity contract
contract_source_code = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelStorage {
    mapping(uint256 => string) private modelHashes;
    uint256 public modelCount;

    function setModelHash(string memory _hash) public {
        modelHashes[modelCount] = _hash;
        modelCount++;
    }

    function getModelHash(uint256 index) public view returns (string memory) {
        require(index < modelCount, "Model does not exist");
        return modelHashes[index];
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

print(f'Contract deployed at address: {contract_address}')
