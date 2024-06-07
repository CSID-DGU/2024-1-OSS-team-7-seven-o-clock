import sys, os
from celery import shared_task
from PIL import Image
from django.conf import settings
import json, pickle


# 재식별을 위한 파이썬 파일
sys.path.append('/root/amd/reid_model')
from reid_model.projects.InterpretationReID.general_evaluation import regist_new_dataset
from reid_model.projects.InterpretationReID.final_total_filtering_jy_multi_process import get_reid_result_top10

sys.path.append('/root/amd/yolo')
from yolo.yolo_twofunction import yolo_vid

# 아래는 /root/amd/yolo/myYolo.py 에서 yolo_vid 메소드를 임포트 하는 방법
# from yolo.myYolo import yolo_vid

def load_data_from_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

@shared_task(bind=True)
def start_re_id_task(self, dataset_name: str, query_img_path: str) -> None:
    # 작업 초기 상태 설정
    self.update_state(state='PROGRESS', meta={'progress': 0, 'total': 100})
    # gallery feat_attr 가져오기
    gallery_feat_attr_pkl_save_path = '/root/amd/reid_model/datasets/' + dataset_name + '/meta/feature_attr.pkl'
    try:
        gallery_feat_attr_dict = load_data_from_pkl(gallery_feat_attr_pkl_save_path)
    except:
        print("데이터셋 등록이 정상적으로 안 되어 있음.")
        exit()
    # query img의 feat, attr 구하기
    

    args = "--config-file /root/amd/reid_model/projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip_eval_only.yml --eval-only  MODEL.DEVICE \"cuda:0\" "
    query_feat_attr_dict = {}
    try:
        query_feat_attr_dict = regist_new_dataset(args=args, dataset_path=query_img_path, save_path=None, task_id=None)
    except:
        print("데이터셋 등록 후 feat, attr 계산 중 오류 발생")
        exit()
    
    reid_result_img_path_list = get_reid_result_top10(query_features_attr=query_feat_attr_dict, gallery_features_attr=gallery_feat_attr_dict, task_id=self.request.id)

    self.update_state(state='SUCCESS', result=reid_result_img_path_list, meta={'progress': 100, 'total': 100})
    return None

@shared_task(bind=True)
def register_dataset(self, dataset_name: str, video_path: str) -> bool:
    task_id = self.request.id
    # 작업 초기 상태 설정
    self.update_state(state='PROGRESS', meta={'progress': 0, 'total': 100})

    dataset_names_dict = {}
    with open(os.path.join(settings.BASE_DIR, 'datasets.json')) as f:
        dataset_names_dict = json.load(f)

    # 데이터셋 이름이 존재하는지 확인
    if dataset_name in dataset_names_dict:
        return False
    
    dataset_save_path = '/root/amd/reid_model/datasets/' + dataset_name + "/imgs"
    
    # yolo를 이용하여 동영상에서 이미지를 추출하는 함수 추가
    try: 
        yolo_vid(dataset_name = dataset_name, video_path = video_path, dataset_save_path = dataset_save_path, task_id = task_id)
        pass
    except:
        print("영상으로 데이터셋을 만드는 과정에서 오류 발생")
        exit()

    dataset_names_dict[dataset_name] = ''

    # 데이터셋 관련 정보를 저장
    with open(os.path.join(settings.BASE_DIR, 'datasets.json'), 'w') as f:
        json.dump(dataset_names_dict, f, ensure_ascii=False)

    # 임시 디렉토리에 저장된 파일을 삭제
    os.remove(video_path)

    # 새로 입력받은 데이터셋의 사진의 feature, attribute를 저장하는 pkl 파일 만들기
    args = "--config-file /root/amd/reid_model/projects/InterpretationReID/configs/Market1501_Circle/circle_R50_ip_eval_only.yml --eval-only  MODEL.DEVICE \"cuda:0\" "
    feat_attr_pkl_save_path = '/root/amd/reid_model/datasets/' + dataset_name + '/meta/feature_attr.pkl'
    try:
        regist_new_dataset(args=args, dataset_path=dataset_save_path, save_path=feat_attr_pkl_save_path, task_id=task_id)
    except:
        print("데이터셋 등록 후 feat, attr 계산 중 오류 발생")
        exit()

    self.update_state(state='SUCCESS', result=True, meta={'progress': 100, 'total': 100})
    return True