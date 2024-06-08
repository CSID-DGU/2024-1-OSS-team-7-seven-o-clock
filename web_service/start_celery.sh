#!/bin/bash

# 에러 발생 시 스크립트 종료
set -e

# Check if Redis server is already running
if pgrep -x "redis-server" > /dev/null
then
    echo "Redis server is already running"
else
    echo "Starting Redis server..."
    redis-server /etc/redis/redis.conf &
    if [ $? -ne 0 ]; then
        echo "Failed to start Redis server"
        exit 1
    fi
fi

# Check if Celery worker is already running
if pgrep -f "celery -A web_demo worker --uid=celeryuser" > /dev/null
then
    echo "Celery worker is already running"
else
    echo "Starting Celery worker..."
    celery -A web_demo worker --loglevel=info --pool=threads &
    if [ $? -ne 0 ]; then
        echo "Failed to start Celery worker"
        exit 1
    fi
fi


# Check if Celery worker is already running
if pgrep -f "python3 manage.py runserver 8080" > /dev/null
then
    echo "django server already started"
else
    echo "Starting django server 8080 ..."
    python3 manage.py runserver 8080 &
    if [ $? -ne 0 ]; then
        echo "Failed to start Celery worker"
        exit 1
    fi
fi
