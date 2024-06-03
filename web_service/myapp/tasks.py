from celery import shared_task
from PIL import Image
import os

@shared_task
def example_task():
    return 'Task executed'

@shared_task
def add(x, y):
    return x + y

@shared_task
def start_re_id_task(dataset: str, image_path: str) -> None:
    image = Image.open(image_path)

    # 재식별 코드


    os.remove(image_path)
    return {"success": True} 

@shared_task
def register_dataset(dataset: str, video_path: str) -> None:
    # 데이터셋 등록

    os.remove(video_path)

    return {"success": True} 