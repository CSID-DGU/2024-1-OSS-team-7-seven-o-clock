document.getElementById('reidentify-btn').addEventListener('click', function () {
    document.getElementById('reidentify-section').classList.remove('hidden');
    document.getElementById('create-dataset-section').classList.add('hidden');
    fetchDatasets();
});

document.getElementById('create-dataset-btn').addEventListener('click', function () {
    document.getElementById('create-dataset-section').classList.remove('hidden');
    document.getElementById('reidentify-section').classList.add('hidden');
});

document.getElementById('drop-zone').addEventListener('click', function () {
    document.getElementById('query-upload').click();
});

document.getElementById('dataset-drop-zone').addEventListener('click', function () {
    document.getElementById('dataset-upload').click();
});

function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.classList.add('dragover');
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (e.target.id === 'drop-zone') {
        document.getElementById('query-upload').files = files;
        handleFileUpload(files[0], 'query-upload');
    } else if (e.target.id === 'dataset-drop-zone') {
        document.getElementById('dataset-upload').files = files;
        handleFileUpload(files[0], 'dataset-upload');
    }
}

function handleFileUpload(file, inputId) {
    const reader = new FileReader();
    reader.onload = function (e) {
        if (inputId === 'query-upload') {
            const dropZone = document.getElementById('drop-zone');
            dropZone.innerHTML = `<img src="${e.target.result}" style="max-height: 100%; max-width: 100%;"/>`;
        } else if (inputId === 'dataset-upload') {
            document.getElementById('file-name').textContent = file.name;
            document.getElementById('file-name').classList.remove('hidden');
        }
    };
    reader.readAsDataURL(file);
}

document.getElementById('drop-zone').addEventListener('dragenter', handleDragEnter);
document.getElementById('drop-zone').addEventListener('dragover', function (e) {
    e.preventDefault();
});
document.getElementById('drop-zone').addEventListener('dragleave', handleDragLeave);
document.getElementById('drop-zone').addEventListener('drop', function (e) {
    e.preventDefault();
    const files = e.dataTransfer.files;
    handleFileUpload(files[0], 'query-upload');
});

document.getElementById('dataset-drop-zone').addEventListener('dragenter', handleDragEnter);
document.getElementById('dataset-drop-zone').addEventListener('dragover', handleDragOver);
document.getElementById('dataset-drop-zone').addEventListener('dragleave', handleDragLeave);
document.getElementById('dataset-drop-zone').addEventListener('drop', handleDrop);

document.getElementById('query-upload').addEventListener('change', function (event) {
    const file = event.target.files[0];
    handleFileUpload(file, 'query-upload');
});

document.getElementById('dataset-upload').addEventListener('change', function (event) {
    const file = event.target.files[0];
    handleFileUpload(file, 'dataset-upload');
});

document.getElementById('start-reid-btn').addEventListener('click', async function () {
    const datasetSelect = document.getElementById('dataset-select').value;
    const queryUpload = document.getElementById('query-upload').files[0];

    if (!queryUpload) {
        alert('쿼리 이미지를 업로드해주세요.');
        return;
    }

    document.getElementById('progress-bar-reid').classList.remove('hidden');

    const formData = new FormData();
    formData.append('query_file', queryUpload);
    formData.append('dataset_name', datasetSelect);

    try {
        const response = await fetch('https:///localhost:8080/start_re_id', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            const taskId = result.task_id;
            console.log('Task ID:', taskId);
            // Handle the response (e.g., polling for results using task_id)
        } else {
            alert('재식별 실패.');
        }
    } catch (error) {
        console.error('Error:', error);
    } finally {
        document.getElementById('progress-bar-reid').classList.add('hidden');
    }
});

document.getElementById('create-dataset-btn-2').addEventListener('click', async function () {
    const datasetUpload = document.getElementById('dataset-upload').files[0];
    const datasetName = document.getElementById('dataset-name').value;

    if (!datasetName) {
        alert('데이터셋의 이름을 입력해주세요.');
        return;
    }

    if (!datasetUpload) {
        alert('파일을 업로드해주세요.');
        return;
    }

    document.getElementById('progress-bar-dataset').classList.remove('hidden');

    const formData = new FormData();
    formData.append('dataset_base', datasetUpload);
    formData.append('dataset_name', datasetName);

    try {
        const response = await fetch('https:///localhost:8080/regist_dataset', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            const taskId = result.task_id;
            console.log('Task ID:', taskId);
            alert('데이터셋 생성 완료.');
        } else {
            alert('데이터셋 생성 실패.');
        }
    } catch (error) {
        console.error('Error:', error);
    } finally {
        document.getElementById('progress-bar-dataset').classList.add('hidden');
    }
});

async function fetchDatasets() {
    try {
        const response = await fetch('https:///localhost:8080/get-datasets', {
            method: 'GET'
        });

        if (response.ok) {
            const datasets = await response.json();
            const datasetSelect = document.getElementById('dataset-select');
            datasetSelect.innerHTML = '';
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.id;
                option.textContent = dataset.name;
                datasetSelect.appendChild(option);
            });
        } else {
            alert('데이터셋 불러오기 실패.');
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

async function fetchMainPage() {
    try {
        const response = await fetch('https:///localhost:8080/regist_page', {
            method: 'GET'
        });

        if (response.ok) {
            const mainPageData = await response.json();
            // 메인 페이지 데이터 처리
        } else {
            alert('메인 페이지 불러오기 실패.');
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// 페이지 로드 시 메인 페이지 데이터 불러오기
document.addEventListener('DOMContentLoaded', function () {
    fetchMainPage();
});
