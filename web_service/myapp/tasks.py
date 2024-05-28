from celery import shared_task

@shared_task
def example_task():
    return 'Task executed'

@shared_task
def add(x, y):
    return x + y