#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
import pickle, csv
import pandas as pd

sys.path.append('.')
os.chdir("/root/amd/reid_model") #/home/workspace/로 이동하는것 방지 

from fastreid.config import get_cfg
from projects.InterpretationReID.interpretationreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from projects.InterpretationReID.interpretationreid.evaluation import ReidEvaluator_General, ReidEvaluator
import projects.InterpretationReID.interpretationreid as PII
from fastreid.modeling.meta_arch import build_model
import torch
from fastreid.utils.logger import setup_logger

class Trainer(DefaultTrainer):
    def __init__(cls, cfg, dataset_path):
        cls.dataset_path = dataset_path
        super.__init__(cfg)

    @classmethod
    def build_model(cls, cfg, dataset_path):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        cls.dataset_path = dataset_path
        model = build_model(cfg)
        return model    
    @classmethod
    def build_test_loader(cls, cfg, test_items):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.

        """

        '''
        원래 dataset_name 을 이용해서 add_build_reid_test_loader -> Market1501_Interpretation을 콜하는 방식인데
        dataset_name 대신 dataset_path 를 이용해서 여기서 로드하는 방식으로 바꾸기.
        '''

        return PII.add_build_reid_test_loader_general(cfg, test_items)
        
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # 현재 원래 코드와 동일한 상태.
        # output vector와 attribute 평가를 따로 저장하는 함수.
        # 저장해놓은 vector와 새로운 쿼리랑 비교해서 결과 뽑아주는 함수 만들어야함.
        
        # return ReidEvaluator(cfg, num_query)
        return ReidEvaluator_General(cfg, num_query)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        img_dataset = regist_dataset(cls.dataset_path)
        data_loader, num_query , name_of_attribute = cls.build_test_loader(cfg, img_dataset)
        evaluator = cls.build_evaluator(cfg, num_query=num_query)
        # model을 평가 모드로 전환
        model.eval()

        data_dict = {}

        total = len(data_loader)
        num_warmup = min(5, total - 1)
        total_compute_time = 0
        with torch.no_grad():
            for idx, inputs in enumerate(data_loader):
                if idx == num_warmup:
                    # celery update 코드가 들어갈 공간
                    total_compute_time = 0

                idx += 1
                outputs_dict = model(inputs)
                # feature 
                outputs = outputs_dict["outputs"]
                attrs = outputs_dict['att_heads']['cls_outputs'].clone().detach()
                # 아래는 0.5가 넘는 경우 1로 판단하도록 만들고 append 하는 코드
                # att_prec = torch.where(attrs>0.5,torch.ones_like(attrs),torch.zeros_like(attrs)).cpu()
                
                #result.append((inputs["img_paths"], outputs.cpu(), att_prec.cpu()))
                img_paths = inputs["img_paths"]
                len_img_paths = len(img_paths)

                for i in range(len_img_paths):
                    # data_dict[img_paths[i]] = [outputs.cpu()[i], att_prec.cpu()[i]]
                    data_dict[img_paths[i]] = [outputs.cpu()[i], attrs.cpu()[i]]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        return data_dict
        

            
def save_data_as_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def load_data_from_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data



def regist_dataset(dataset_path):
    pid = 0
    camid = 0
    p_attr = torch.full((26,), 0)
    # ( '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/bounding_box_test/0330_c5s1_075798_04.jpg', 330, 4, tensor([-1., -1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.,  1., 1., -1.,  1.,  1., -1., -1., -1., -1., -1.,  1., -1., -1.]) )
    dataset = []
    img_paths = []
    print(f"dataset_path: {dataset_path}")
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            img_paths.append(filename)
    for img_path in img_paths:
        dataset.append((dataset_path + "/" + img_path, pid, camid, p_attr))
    return dataset


def setup(args):
    """
    Create configs_old and perform basic setups.
    """
    cfg = get_cfg()
    PII.add_interpretation_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def regist_new_dataset(args, dataset_path, save_path):
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = Trainer.build_model(cfg, dataset_path)

    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        
    res = Trainer.test(cfg, model)
    save_data_as_pkl(res, save_path)
    return res

def filter_gallery_using_attr(query_pkl_path, gallery_pkl_path, dataset_path):
    print()
    



def get_feat_dist_query_gallery(args, dataset_path, query_pkl_path, gallery_pkl_path, save_path):
    query_features_attr = load_data_from_pkl(query_pkl_path)
    gallery_features_attr = load_data_from_pkl(gallery_pkl_path)
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = Trainer.build_model(cfg, dataset_path)
    evaluator = Trainer.build_evaluator(cfg, num_query= 0)

    # 각 쿼리와 갤러리 이미지의 이름
    query_keys = list(query_features_attr.keys())
    gallery_keys = list(gallery_features_attr.keys())

    query_features = [value[0] for value in query_features_attr.values()]
    gallery_features = [value[0] for value in gallery_features_attr.values()]
    # print(query_features)
    query_features = torch.stack(query_features)
    gallery_features = torch.stack(gallery_features)

    # dist: query_feat의 i번째 벡터와 gallery_feat의 j번째 벡터의 거리 값
    dist = evaluator.cal_dist(cfg.TEST.METRIC, query_features, gallery_features)
    
    dist_df = pd.DataFrame(dist, index=query_keys, columns=gallery_keys)

    # save_data_as_pkl(dist, save_path)
    dist_df.to_pickle(save_path)
    # dist_df.to_csv("/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/meta/query_gallery_feature_distance/query_gallery_feature_distance_10.csv")


    # 모델에 사진 하나 넣고 벡터, 속성 평가 값만 받아오도록 만들고
    # 그걸 리턴해줌. (샐러리 task로)
    # 샐러리 task에서는 데이터베이스 조회해서 갤러리에 있는 애들과 dist 측정하고 
    # 그 중 가장 유사한 10개 뱉어줌.

def get_result_using_dist_filtered_gallery(gallery_path, dist_path):
    result = ""
    query_per_gallery_dict = load_data_from_pkl(gallery_path)
    # query_per_gallery_dict의 key와 value 리스트에 basename 적용
    query_per_gallery_dict = {
        os.path.basename(query_img_path).strip(): [os.path.basename(gallery_img_path).strip() for gallery_img_path in gallery_img_path_list]
        for query_img_path, gallery_img_path_list in query_per_gallery_dict.items()
    }
    dist_df = pd.read_pickle(dist_path)
    # 인덱스와 컬럼을 순수 파일 이름으로 변환
    dist_df.index = dist_df.index.map(lambda x: os.path.basename(x).strip())
    dist_df.columns = dist_df.columns.map(lambda x: os.path.basename(x).strip())

    # query_img_name = query_img_name.strip()
    # gallery_img_name_list = [img.strip() for img in gallery_img_name_list]

    total_count = 0
    rank1 = rank5 = rank10 = 0 
    for query_img_name, gallery_img_name_list in query_per_gallery_dict.items():
        total_count+=1
                  
        top10_list = dist_df.loc[query_img_name, gallery_img_name_list].sort_values().index.tolist()[0:11]
        print(top10_list)
        query_pid = query_img_name[:4]
        try:
            if query_pid == top10_list[0][:4]:
                rank1 +=1
        except:
            continue
        for i in range(5):
            try:
                if query_pid == top10_list[i][:4]:
                    rank5 +=1
            except:
                continue
        for i in range(10):
            try:
                if query_pid == top10_list[i][:4]:
                    rank10 +=1
            except:
                continue
    result = [rank1/total_count, rank5/total_count, rank10/total_count]
    save_path = "/root/amd/reid_model/projects/InterpretationReID/filtering_result/result.txt"
    with open(save_path, 'w') as file:
                file.write(f"top1,1 resolution100 rank1: {rank1/total_count}, rank5: {rank5/total_count}, rank10: { rank10/total_count}")

    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    gallery_pkl_path = "/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/meta/gallery_feature_attr.pkl"
    for i in range(1, 11):
        query_pkl_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/query/meta/query_{i* 10}_feature_attr.pkl"
        save_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/meta/query_gallery_feature_distance/query_gallery_feature_distance_{i*10}.pkl"
        get_feat_dist_query_gallery(args, dataset_path= "", query_pkl_path=query_pkl_path, gallery_pkl_path=gallery_pkl_path, save_path=save_path)
    

    #dataset_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/bounding_box_test'
    #save_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/meta/gallery_feature_attr.pkl'
    #regist_new_dataset(args, dataset_path=dataset_path, save_path=save_path)

    #for i in range(1, 11):
        #dataset_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/query/query_{i*10}"
        #save_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/query/meta/query_{i*10}_feature_attr.pkl"
        #regist_new_dataset(args, dataset_path=dataset_path, save_path=save_path)
    for i in range(10, 11):
        gallery_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/top_up1_down1/filtered_gallery/meta/remain_gallery_path_list_per_query_img_{i*10}.pkl"
        dist_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/meta/query_gallery_feature_distance/query_gallery_feature_distance_{i*10}.pkl"
        get_result_using_dist_filtered_gallery(gallery_path=gallery_path, dist_path=dist_path)