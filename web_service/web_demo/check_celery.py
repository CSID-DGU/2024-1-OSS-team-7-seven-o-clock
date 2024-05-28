# check_celery.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_demo.celery_app import app

@app.task
def test_task():
    return 'Test task executed successfully!'

if __name__ == '__main__':
    result = test_task.delay()
    print(f'Task result: {result.get(timeout=10)}')
