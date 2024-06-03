from typing import cast
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from myapp import tasks
from web_demo.celery_app import app
from django.http import HttpRequest
from django.views.decorators.csrf import csrf_exempt
from myapp.forms import StartReIdForm, RegistDataset
from myapp.uploadedfile import TemporaryUploadedFile
from myapp.uploadhandler import TemporaryFileUploadHandler

def main_view(request):
    async_result = tasks.add.delay('dataset', '')
    return render(request, 'main.html', {'task_id': async_result.id})

@csrf_exempt
def start_re_id(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({ }, status=405)
    
    request.upload_handlers = [TemporaryFileUploadHandler(request=request)]

    form = StartReIdForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({}, status=400)
    
    dataset = form.cleaned_data['dataset']

    image = cast(TemporaryUploadedFile, form.cleaned_data['image']).temporary_file_path()

    async_result = tasks.start_re_id_task.delay(dataset, image)
    return JsonResponse({ "id": async_result.id })

@csrf_exempt
def regist_dataset(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({ }, status=405)
    
    request.upload_handlers = [TemporaryFileUploadHandler(request=request)]

    form = RegistDataset(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({}, status=400)

    dataset = form.cleaned_data['dataset']

    video = cast(TemporaryUploadedFile, form.cleaned_data['video']).temporary_file_path()

    async_result = tasks.register_dataset.delay(dataset, video)
    return JsonResponse({ "id": async_result.id })
    
@csrf_exempt
def get_state(request, task_id):
    async_result = app.AsyncResult(task_id)
    if async_result.failed():
        return JsonResponse({ "status": "FAILED" })
    
    result = None

    if async_result.ready():
        result = async_result.get()

    return JsonResponse({ "status": async_result.status, "result": result })


