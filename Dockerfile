FROM junyoungim/amd-final:latest

WORKDIR /root/amd

RUN git pull origin main

CMD ["/root/amd/web_service/start_celery.sh"]

# docker build -t <imageName> .
# docker run --gpus all --name <containerName> -d <imageName>
# 도커 컨테이너에 들어가서 apt install intel-mkl, pip3 install faiss-cpu
