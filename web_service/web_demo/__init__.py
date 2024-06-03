# web_demo/__init__.py
from __future__ import absolute_import, unicode_literals
import os
import django

from .celery_app import app as celery_app



__all__ = ('celery_app',)
