from celery import shared_task
from PIL import Image
import os
from django.conf import settings
import json


@shared_task
def example_task():
    return 'Task executed'

@shared_task
def add(x, y):
    return x + y

@shared_task
def start_re_id_task(dataset: str, image_path: str) -> None:
    # 파일을 Pillow 로 불러오기
    image = Image.open(image_path)
 
    # 재식별 내용

    return None

@shared_task
def register_dataset(dataset: str, video_path: str) -> bool:
    datasets = {}
    with open(os.path.join(settings.BASE_DIR, 'datasets.json')) as f:
        datasets = json.load(f)

    # 데이터셋 이름이 존재하는지 확인
    if dataset in datasets:
        return False
    
    # 데이터셋 처리 내용

    datasets[dataset] = ''

    # 데이터셋 관련 정보를 저장
    with open(os.path.join(settings.BASE_DIR, 'datasets.json'), 'w') as f:
        json.dump(datasets, f, ensure_ascii=False)

    # 임시 디렉토리에 저장된 파일을 삭제
    os.remove(video_path)

    return True