#!/bin/bash
ps -aux | grep 'celery' | grep -v 'grep' | awk '{print $2}' | xargs -r kill -9

ps -aux | grep 'manage.py runserver' | grep -v 'grep' | awk '{print $2}' | xargs -r kill -9

ps -aux | grep 'redis-server' | grep -v 'grep' | awk '{print $2}' | xargs -r kill -9

echo "Celery, Django, Redis 관련 프로세스를 종료했습니다."

