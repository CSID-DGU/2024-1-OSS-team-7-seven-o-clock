from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from myapp.tasks import add
import os

def main_view(request):
    # 비동기적으로 Celery 태스크를 실행
    result = add.delay(4, 6)
    return render(request, 'main.html', {'task_id': result.id})

def get_result(request, task_id):
    # Celery 태스크의 결과를 가져옴
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    if result.ready():
        return JsonResponse({'status': 'SUCCESS', 'result': result.result})
    else:
        return JsonResponse({'status': 'PENDING'})