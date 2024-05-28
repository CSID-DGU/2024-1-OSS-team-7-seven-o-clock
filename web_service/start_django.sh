python3 manage.py runserver 8080 &

celery -A web_demo worker --uid=celeryuser
