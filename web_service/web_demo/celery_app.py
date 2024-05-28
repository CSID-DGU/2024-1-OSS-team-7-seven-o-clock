# web_demo/celery_app.py
from __future__ import absolute_import, unicode_literals
from celery import Celery
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web_demo.settings')

app = Celery('web_demo')

app.conf.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
)

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()
