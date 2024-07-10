### 프로젝트 실행 방법
1. 깃허브 프로젝트에 포함된 도커파일 다운로드.
2. 도커 파일을 다운로드 받은 위치를 powershell 또는 terminal로 열고, 'docker build -t oss7 .' 명령어 입력
3. 'docker run --gpus all -m 128G --memory-swap 200G -p 8080:8080 -it --name test-oss7 oss7' 명령어를 입력하여 도커 컨테이너 실행
4. 브라우저를 열고 주소창에 localhost:8080 을 입력하면 메인 페이지에 접속할 수 있음.
