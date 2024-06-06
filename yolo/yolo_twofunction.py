#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#여기는 만약 필요하다면!!
"""
!pip install ultralytics
!pip install onnxruntime-gpu

from ultralytics import YOLO
import zipfile
import cv2
import os
from PIL import Image

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model.to('cuda')
# Export the model to ONNX format
model.export(format='onnx')  # creates 'yolov8n.onnx'

# Load the exported ONNX model
onnx_model = YOLO('yolov8n.onnx')
"""


#몇 퍼센트 됐는지 확인하는 법 --
                    #vid 함수 : total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ->전체 프레임 수 이거 아래 코드에 없어서 쓰려면 추가해야해요
                    # -> frame_count / total_frames로 구함. frame_count는 이미 있는거라 추가로 정의할 필요 없습니다!
                    
                    #zip 함수
                    #image_paths -> 여기에 압축풀기된 이미지의 경로가 저장됨(리스트)
                    # frmae_count / len(image_paths)로 구


# In[1]:


from ultralytics import YOLO
import zipfile
import cv2
import os, redis, json, datetime
from PIL import Image

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
model.to('cuda')
# Export the model to ONNX format
model.export(format='onnx')  # creates 'yolov8n.onnx'

# Load the exported ONNX model
onnx_model = YOLO('yolov8n.onnx')


# In[2]:

# redis client를 싱글톤으로 관리
def create_redis_client():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
            # 연결 테스트
            redis_client.ping()
            print("Redis 클라이언트 연결 성공")
        except Exception as e:
            print("Redis 클라이언트 시작 실패:", e)
            redis_client = None
            exit()
    return redis_client

def yolo_vid(dataset_name: str, video_path: str, dataset_save_path: str, task_id):
    redis_client = create_redis_client()
    task_key = f'celery-task-meta-{task_id}'

    try:
        task_data = redis_client.get(task_key)
    except:
        print("레디스에서 해당 task 가져오기 실패")
        exit()
    try:
        task_data = json.loads(task_data)
    except:
        print("task data json 파싱 실패")
        exit()

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 만약 몇 퍼센트됐는지 이 함수 내에서 구하려 한다면 이 코드를 작성
    absolute_paths = []
    idx = 0
    
    update_interval = total_frames // 50
    progress = 0

    crop_dir_name = os.path.join(dataset_save_path, dataset_name) #저장폴더 만들기
    if not os.path.exists(crop_dir_name):
        os.makedirs(crop_dir_name) 

    frame_count = 0 
    while cap.isOpened():#이미지 한장한장 뽑아오면서 돌리기
        success, im0 = cap.read()
        if not success:
            break
            
        if frame_count % 50 == 0: #50프레임마다/ 이건 숫자 바꿔도 됩니다/
            results = onnx_model(im0, classes=0, conf=0.4) #conf = 0.4 이거는 40프로 이상인 것만 가져온다는 건데 수치 바꿔도 돼요
            boxes = results[0].boxes.xyxy.tolist()  

            for box in boxes: #잘라서 저장
                idx += 1
                crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                image_path = os.path.join(crop_dir_name, f"{dataset_name}_{idx}.png")
                cv2.imwrite(image_path, crop_obj)
                absolute_path = os.path.abspath(image_path)
                absolute_paths.append(absolute_path)
            #frame_count / total_frames 이쯤에 이거 넣으면 지금 얼마나 했는지 구합니다
        frame_count += 1
        if frame_count % update_interval == 0:
            # celery task status update 코드
            progress += 1
            task_data['date_done'] = datetime.now(datetime.UTC).isoformat()
            task_data['meta']['progress'] = progress
            redis_client.set(task_key, json.dumps(task_data))

    cap.release()
    cv2.destroyAllWindows()

    # 동영상 처리 완료 후 진행도 50으로 설정
    progress = 50
    task_data['date_done'] = datetime.now(datetime.UTC).isoformat()
    task_data['meta']['progress'] = progress
    redis_client.set(task_key, json.dumps(task_data))

    return absolute_paths

def yolo_zip(dataset_name: str, zip_file_path: str, dataset_save_path: str):
    absolute_paths = []

    crop_dir_name = os.path.join(dataset_save_path, dataset_name)
    if not os.path.exists(crop_dir_name):
        os.makedirs(crop_dir_name)
    
    if not os.path.exists('zip_extract'): #압축풀기
        os.makedirs('zip_extract')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('zip_extract')
    image_paths = []

    for root, _, files in os.walk('zip_extract'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                absolute_path = os.path.abspath(os.path.join(root, file))
                image_paths.append(absolute_path)   #여기까지 압축풀기
    
    frame_count = 0 
    for imgpath in image_paths:
        im0 = Image.open(imgpath)
        if frame_count%1 == 0 : #이건 모든 사진을 쓰는거라 frame_count를 1로 했어요. 만약 사진 2장 당 하나 쓰려면 2 이런식으로 조절 가능
            results = onnx_model(im0, classes=0, conf = 0.4)
            boxes = results[0].boxes.xyxy.tolist()  
            
            for idx, box in enumerate(results[0].boxes.xyxy.tolist()): #사진 저장
                x1, y1, x2, y2 = box
                img_cropped = im0.crop((x1,y1,x2,y2))
                img_cropped.save(f"{dataset_save_path}/{dataset_name}/{dataset_name}_{idx}.png")
                absolute_path = os.path.abspath(f"{dataset_save_path}/{dataset_name}/{dataset_name}_{idx}.png")
                absolute_paths.append(absolute_path)
        
        #frmae_count / len(image_paths) 이쯤에 넣으면 얼마나 완료했는지 구합니다
        frame_count+=1
        
    return absolute_paths

