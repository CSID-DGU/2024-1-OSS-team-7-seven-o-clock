from typing import cast
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from myapp import tasks
from web_demo import settings
from web_demo.celery_app import app
from django.http import HttpRequest
from django.views.decorators.csrf import csrf_exempt
from myapp.forms import StartReIdForm, RegistDataset
from myapp.uploadedfile import TemporaryUploadedFile
from myapp.uploadhandler import TemporaryFileUploadHandler
from django.core.files.storage import FileSystemStorage
import os
import shutil
from django.conf import settings
import json
from datetime import datetime
from pathlib import Path

def main_view(request):
    return render(request, 'index.html')

def get_datasets(request):
    dataset_names_dict = {}
    with open(os.path.join(settings.BASE_DIR, 'datasets.json')) as f:
        dataset_names_dict = json.load(f)
    dataset_names = None
    try: 
        dataset_names = json.dumps(list(dataset_names_dict.keys()))
    except:
        print("json dumps err")
    dataset_names = list(dataset_names_dict.keys())
    return JsonResponse({'datasetList': dataset_names}, status=200)

@csrf_exempt
def start_re_id(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({ }, status=405)
    


    image_path = None
    form = StartReIdForm(request.POST, request.FILES)
    if form.is_valid():
        # FileSystemStorage를 사용하여 파일을 저장합니다.
        query_file = form.cleaned_data['query_file']
        fs = FileSystemStorage()
        filename = fs.save(str(datetime.now())+query_file.name, query_file)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
    else: 
        return JsonResponse({}, status=400)
    # 데이터셋 이름을 가져옴
    dataset = form.cleaned_data['dataset_name']

    '''
    # 이미지가 저장된 임시 경로를 가져옴
    tmp_image_path = cast(TemporaryUploadedFile, form.cleaned_data['query_file']).temporary_file_path()

    # 파일 확장자를 가져옴
    file_name, ext = os.path.splitext(tmp_image_path)

    # 새롭게 저장할 경로를 지정
    image_path = os.path.join(settings.MEDIA_ROOT, 'start_re_id', f"{file_name}{ext}")

    # 파일을 현재 경로로 이동
    shutil.move(tmp_image_path, image_path)
    '''

    # 작업 요청
    async_result = tasks.start_re_id_task.delay(dataset, image_path)

    return JsonResponse({ "task_id": async_result.id })

@csrf_exempt
def regist_dataset(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({ }, status=405)
    
    request.upload_handlers = [TemporaryFileUploadHandler(request=request)]
    print("Request POST data:", request.POST)
    print("Request FILES data:", request.FILES)
    # 데이터 전처리를 진행
    form = RegistDataset(request.POST, request.FILES)
    if not form.is_valid():
        print("FORM ERROR", form.errors)
        return JsonResponse({}, status=400)
    
    datasets = {}
    with open(os.path.join(settings.BASE_DIR, 'datasets.json')) as f:
        datasets = json.load(f)
    
    # 데이터셋 이름을 가져옴
    dataset = form.cleaned_data['dataset_name']

    # 데이터셋 이름이 존재하는지 확인
    if dataset in datasets:
        return JsonResponse({ "success": False, "message": "이미 존재하는 데이터셋 이름입니다" }, status=409)

    # 영상이 저장된 임시 경로를 가져옴
    video = cast(TemporaryUploadedFile, form.cleaned_data['dataset_base']).temporary_file_path()

    # 작업 요청
    async_result = tasks.register_dataset.delay(dataset, video)
    return JsonResponse({ "task_id": async_result.id })
    
@csrf_exempt
def get_state(request, task_id):
    # task_id 로 AsyncResult 를 가져옴
    async_result = app.AsyncResult(task_id)
    if async_result.failed():
        return JsonResponse({ "status": "FAILED" })
    
    result = None
    progress = 0
    total = 100

    # 작업이 준비된 경우 결과를 가져옴
    if async_result.ready():
        result = async_result.get()
        print(f"작업이 완료됨: {result}")
    elif async_result.status == 'PROGRESS':
        progress = async_result.info.get('progress', 0)
        total = async_result.info.get('total', 100)

    return JsonResponse({
        "state": async_result.status,
        "result": result,
        "progress": progress,
        "total": total
    })


