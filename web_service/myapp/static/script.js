document.addEventListener('DOMContentLoaded', function () {
    const startReidBtn = document.getElementById('start-reid-btn');
    const queryUpload = document.getElementById('query-upload');
    const datasetSelect = document.getElementById('dataset-select');
    const progressBarReid = document.getElementById('progress-bar-reid');
    const resultReid = document.getElementById('result-reid');
    const createDatasetBtn = document.getElementById('create-dataset-btn-2');
    const datasetUpload = document.getElementById('dataset-upload');
    const progressBarDataset = document.getElementById('progress-bar-dataset');

    if (!queryUpload || !datasetSelect || !startReidBtn || !progressBarReid || !resultReid || !createDatasetBtn || !datasetUpload || !progressBarDataset) {
        console.error('필요한 요소를 찾을 수 없습니다.');
        return;
    }

    document.getElementById('reidentify-btn').addEventListener('click', function () {
        document.getElementById('reidentify-section').classList.remove('hidden');
        document.getElementById('create-dataset-section').classList.add('hidden');
    });

    document.getElementById('create-dataset-btn').addEventListener('click', function () {
        document.getElementById('create-dataset-section').classList.remove('hidden');
        document.getElementById('reidentify-section').classList.add('hidden');
    });

    document.getElementById('drop-zone').addEventListener('click', function () {
        queryUpload.click();
    });

    document.getElementById('dataset-drop-zone').addEventListener('click', function () {
        datasetUpload.click();
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
            queryUpload.files = files;
            handleFileUpload(files[0], 'query-upload');
        } else if (e.target.id === 'dataset-drop-zone') {
            datasetUpload.files = files;
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
    document.getElementById('drop-zone').addEventListener('dragover', handleDragOver);
    document.getElementById('drop-zone').addEventListener('dragleave', handleDragLeave);
    document.getElementById('drop-zone').addEventListener('drop', handleDrop);

    document.getElementById('dataset-drop-zone').addEventListener('dragenter', handleDragEnter);
    document.getElementById('dataset-drop-zone').addEventListener('dragover', handleDragOver);
    document.getElementById('dataset-drop-zone').addEventListener('dragleave', handleDragLeave);
    document.getElementById('dataset-drop-zone').addEventListener('drop', handleDrop);

    queryUpload.addEventListener('change', function (event) {
        const file = event.target.files[0];
        handleFileUpload(file, 'query-upload');
    });

    datasetUpload.addEventListener('change', function (event) {
        const file = event.target.files[0];
        handleFileUpload(file, 'dataset-upload');
    });

    async function checkTaskStatus(taskId, updateProgress, onSuccess) {
        try {
            const response = await fetch(`http://localhost:8080/get_state?task_id=${taskId}`, {
                method: 'GET'
            });

            if (response.ok) {
                const result = await response.json();
                updateProgress(result.current, result.total);

                if (result.state === 'SUCCESS') {
                    onSuccess(result.result);
                    clearInterval(taskStatusInterval); // Stop checking after success
                } else if (result.state === 'FAILURE') {
                    alert('작업 실패.');
                    clearInterval(taskStatusInterval); // Stop checking after failure
                }
            } else {
                alert('상태 조회 실패.');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }

    function updateProgress(progressBar, current, total) {
        const percentage = (current / total) * 100;
        progressBar.style.width = `${percentage}%`;
    }

    function displayResults(results) {
        resultReid.innerHTML = '';

        results.forEach(result => {
            const img = document.createElement('img');
            img.src = `data:${result.content_type};base64,${result.data}`;
            img.alt = result.name;
            resultReid.appendChild(img);
        });

        resultReid.classList.remove('hidden');
    }

    let taskStatusInterval;

    startReidBtn.addEventListener('click', async function () {
        const datasetValue = datasetSelect.value;
        const queryFile = queryUpload.files[0];

        if (!queryFile) {
            alert('쿼리 이미지를 업로드해주세요.');
            return;
        }

        progressBarReid.classList.remove('hidden');

        const formData = new FormData();
        formData.append('query_file', queryFile);
        formData.append('dataset_name', datasetValue);

        try {
            // FormData 내용을 확인
            for (let pair of formData.entries()) {
                console.log(pair[0] + ', ' + pair[1]);
            }

            const response = await fetch('http://localhost:8080/start_re_id', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                const taskId = result.task_id;

                if (!taskId) {
                    throw new Error('task_id가 응답에 없습니다.');
                }

                console.log('Task ID:', taskId);
                taskStatusInterval = setInterval(() => checkTaskStatus(taskId, (current, total) => {
                    updateProgress(progressBarReid.querySelector('.progress-bar'), current, total);
                    if (current === total) {
                        alert('재식별 완료.');
                        progressBarReid.classList.add('hidden');
                    }
                }, displayResults), 5000); // Check status every 5 seconds
            } else {
                alert('재식별 실패.');
            }
        } catch (error) {
            alert(`에러 발생: ${error.message}`);
            console.error('Error:', error);
        }
    });

    createDatasetBtn.addEventListener('click', async function () {
        const datasetFile = datasetUpload.files[0];
        const datasetName = document.getElementById('dataset-name').value;

        if (!datasetName) {
            alert('데이터셋의 이름을 입력해주세요.');
            return;
        }

        if (!datasetFile) {
            alert('파일을 업로드해주세요.');
            return;
        }

        progressBarDataset.classList.remove('hidden');

        const formData = new FormData();
        formData.append('dataset_base', datasetFile);
        formData.append('dataset_name', datasetName);

        try {
            const response = await fetch('http://localhost:8080/regist_dataset', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                const taskId = result.task_id;

                if (!taskId) {
                    throw new Error('task_id가 응답에 없습니다.');
                }

                console.log('Task ID:', taskId);
                taskStatusInterval = setInterval(() => checkTaskStatus(taskId, (current, total) => {
                    updateProgress(progressBarDataset.querySelector('.progress-bar'), current, total);
                    if (current === total) {
                        alert('데이터셋 생성 완료.');
                        progressBarDataset.classList.add('hidden');
                    }
                }, () => { }), 5000); // Check status every 5 seconds
            } else {
                alert('데이터셋 생성 실패.');
            }
        } catch (error) {
            alert(`에러 발생: ${error.message}`);
            console.error('Error:', error);
        }
    });

    async function fetchDatasets() {
        try {
            const response = await fetch('http://localhost:8080/get-datasets', {
                method: 'GET'
            });

            if (response.ok) {
                const datasets = await response.json();
                const datasetList = datasets.datasetList;

                datasetSelect.innerHTML = '';
                datasetList.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset; // 데이터셋의 ID를 옵션 값으로 설정
                    option.textContent = dataset; // 데이터셋의 이름을 옵션 텍스트로 설정
                    datasetSelect.appendChild(option);
                });
            } else {
                alert('데이터셋 불러오기 실패.');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }

    fetchDatasets(); // 페이지 로드 시 데이터셋 불러오기
});
