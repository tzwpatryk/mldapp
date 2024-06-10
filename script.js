function generateClassInputs() {
    const numClasses = document.getElementById('numClasses').value;
    const classInputs = document.getElementById('classInputs');
    classInputs.innerHTML = '';

    for (let i = 0; i < numClasses; i++) {
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = `Class ${i} Name`;
        input.id = `class${i}`;
        classInputs.appendChild(input);
        classInputs.appendChild(document.createElement('br'));
    }
}

async function uploadModel() {
    const modelInput = document.getElementById('modelInput');
    const file = modelInput.files[0];
    const numClasses = document.getElementById('numClasses').value;
    const classNames = {};

    for (let i = 0; i < numClasses; i++) {
        classNames[i] = document.getElementById(`class${i}`).value;
    }

    if (!file) {
        document.getElementById('uploadResult').innerText = 'No file selected';
        return;
    }

    const formData = new FormData();
    formData.append('model', file);
    formData.append('num_classes', numClasses);
    formData.append('class_names', JSON.stringify(classNames));

    try {
        const response = await fetch('http://localhost:5050/upload_model', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById('uploadResult').innerText = 'Upload successful: ' + result.ipfs_hash;
        loadModels();
    } catch (error) {
        document.getElementById('uploadResult').innerText = 'Error: ' + error.message;
        console.error('Error during model upload:', error);
    }
}

async function loadModels() {
    try {
        const response = await fetch('http://localhost:5050/get_models');
        const models = await response.json();
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '';

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.text = `Model ${model.id}`;
            modelSelect.add(option);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function displayImage(event) {
    const image = document.getElementById('uploadedImage');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.style.display = 'block';
}

async function uploadAndPredict() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const modelId = document.getElementById('modelSelect').value;
    const baseModel = document.getElementById('baseModelSelect').value;

    if (!file) {
        document.getElementById('predictionResult').innerText = 'No file selected';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);
    formData.append('base_model', baseModel);

    try {
        const response = await fetch('http://localhost:5050/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        const classId = result.class_id;
        document.getElementById('predictionResult').innerText = 'Prediction: ' + classId;
    } catch (error) {
        document.getElementById('predictionResult').innerText = 'Error: ' + error.message;
        console.error('Error during prediction:', error);
    }
}

// Load models when the page loads
window.onload = loadModels;
