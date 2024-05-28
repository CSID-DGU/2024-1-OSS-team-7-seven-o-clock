import scipy.io
import numpy as np
import pandas as pd
import torch.nn.functional as F
import time
import pickle
import os
import torch
import shutil
from concurrent.futures import ProcessPoolExecutor
import asyncio


def save_data_as_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
def load_data_from_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

up_color_dict={18:"black",19:"blue",20:"gray",21:"green",22:"purple",23:"red",24:"white",25:"yellow",0:"unknown"} #상의색 8개
down_color_dict={4:"black",5:"blue",6:"brown",7:"gray",8:"green",9:"pink",10:"purple",11:"white",12:"yellow",0:"unknown"} #하의색 9개
color_dict = {4:"black",5:"blue",6:"brown",7:"gray",8:"green",9:"pink",10:"purple",11:"white",12:"yellow", 18:"black",19:"blue",20:"gray",21:"green",22:"purple",23:"red",24:"white",25:"yellow",0:"unknown"}

def load_gallery_real_attribute():
    gallery_real_attribute_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/test_attribute.csv'
    gallery_real_attr_pkl_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/filtered_gallery/meta/gallery_real_attribute.pkl'

    # gallery attr label 을 load하는데, 만약 상의 하의 인덱스로 저장된 것이 있다면 가져오고
    if os.path.exists(gallery_real_attr_pkl_path):
        gallery_real_attribute = load_data_from_pkl(gallery_real_attr_pkl_path)
    # 없다면 새로 만든다.
    else:
        gallery_real_attribute_df = pd.read_csv(gallery_real_attribute_path).set_index('img_name').T.to_dict('list')
        gallery_real_attribute = {}
        for img_name, value in gallery_real_attribute_df.items():
            try:
                upper_true_idx = 18 + value[18:26].index(2)
            except:
                upper_true_idx = 0
                # print(f"up: {img_name}")
            try:
                down_true_idx = 4 + value[4:13].index(2)
            except:
                down_true_idx = 0
                # print(f"down: {img_name}")
            img_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/bounding_box_test/' + img_name
            gallery_real_attribute[img_path] = [upper_true_idx, down_true_idx]
        # print(gallery_real_attribute)
        save_data_as_pkl(gallery_real_attribute, gallery_real_attr_pkl_path)

    gallery_fake_features_attr_pkl_path = '/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/meta/gallery_feature_attr.pkl'
    gallery_fake_features_attr = load_data_from_pkl(gallery_fake_features_attr_pkl_path)
    return gallery_real_attribute, gallery_fake_features_attr

def filtering_preprocess():
    gallery_real_attribute, gallery_fake_features_attr = load_gallery_real_attribute()

    # 삭제할 키를 모아두는 리스트
    junk_img_features_attr = [key for key in gallery_fake_features_attr if os.path.basename(key).startswith('0000') or os.path.basename(key).startswith('-1')]

    # 키 삭제
    for key in junk_img_features_attr:
        del gallery_fake_features_attr[key]
    return gallery_real_attribute, gallery_fake_features_attr

######
def generate_filtered_gallery(iter_num, FILTERING_METHOD = "top", TOP_UP_K = 1, TOP_DOWN_K = 1, THRESHOLD = 0.5 ):
    gallery_real_attribute, gallery_fake_features_attr = filtering_preprocess()
    TP_list=[]
    FP_list=[]
    TN_list=[]
    FN_list=[]
    F1_list=[]

    # 쿼리 이미지에 대해 상, 하의 속성 구하기
    # 일치하는 것만 갤러리에 추가 -> TP, TN, FP, FN 구하기, Rank 구하기 
    # TP, TN, FP, FN, Rank 구한 것들 추가해서 결과 내기
    query_features_attr_pkl_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/query/meta/query_{(iter_num+1) * 10}_feature_attr.pkl"
    query_features_attr = load_data_from_pkl(query_features_attr_pkl_path)
        
    # print(gallery_real_attribute)
    # key: img_path + img_name, value: feature (tensor 2048)
    query_features = {}
    # key: img_path + img_name, value: attr (tensor 26)
    gallery_features = {}

    # key: img_path + img_name, value: feature
    query_attr = {}
    # key: img_path + img_name, value: attr (tensor 26)
    gallery_attr = {}

    # img_path + img_name
    query_img_path_list = query_features_attr.keys()
    gallery_img_path_list = gallery_fake_features_attr.keys()

    def get_topk_colors(attr, start_idx, end_idx, k):
        _, topk_idx = attr[start_idx:end_idx].topk(k)
        return {color_dict[start_idx + idx.item()] for idx in topk_idx}
    # 구현해야함.
    def get_over_n_colors(attr, start_idx, end_idx, threshold):

        return 1
    if torch.cuda.is_available():
        for key, value in query_features_attr.items():
            query_features[key] = value[0].cuda()
            query_attr[key] = value[1].cuda()
        for key, value in gallery_fake_features_attr.items():
            gallery_features[key] = value[0].cuda()
            gallery_attr[key] = value[1].cuda()
    else:
        for key, value in query_features_attr.items():
            query_features[key] = value[0]
            query_attr[key] = value[1]
        for key, value in gallery_fake_features_attr.items():
            gallery_features[key] = value[0]
            gallery_attr[key] = value[1]

    # print(query_attr.keys())
    
    ### 각 쿼리별로 필터링 후 남은 갤러리의 path_list
    # key : query_img_path, value: filtered_gallery_path_list
    remain_gallery_path_list_per_query_img = {}

    i = 0
    # n
    # test
    # query_img_path_list = query_img_path_list[:10]

    # gallery fake up, down color를 이미 연산했다면 다시 연산하지 않도록 만듦.
    # key: gallery img path, value: [ [up_attr], [down_attr] ]
    known_gallery_fake_attr_dict = {}

    for query_img_path in query_img_path_list:
        # print(img_path)
        # m 
        tp = tn = fp = fn = 0
        remain_filtered_gallery_features_dict = {}
        remain_gallery_path_list_per_query_img[query_img_path] = []

        # 해당 쿼리 이미지 상의 속성 값 중 가장 큰 3개의 인덱스를 가져옴
        if FILTERING_METHOD == "top":
            # 모델이 예측한 상의 하의 색 중 가장 확률이 높은 것
            query_fake_up_set = get_topk_colors(attr=query_attr[query_img_path], start_idx=18, end_idx=26, k=TOP_UP_K)
            query_fake_down_set = get_topk_colors(attr=query_attr[query_img_path], start_idx=4, end_idx=13, k=TOP_DOWN_K)
        elif FILTERING_METHOD == "over05":
            query_fake_up_set = get_over_n_colors(attr=query_attr[query_img_path], start_idx=18, end_idx=26, threshold=THRESHOLD)
            query_fake_down_set = get_over_n_colors(attr=query_attr[query_img_path], start_idx=4, end_idx=13, threshold=THRESHOLD)
        # print(query_fake_up_set)

        start_time=time.time()
        for gallery_img_path in gallery_img_path_list:        
            # idx가 18~25, 4~12로 잘 저장되어 있음.
            # 또는 라벨을 결정할 수 없는 경우 0으로 저장되어 있음.

            # 이미 해당 상의/하의 색상을 계산한 적이 없다면
            if gallery_img_path not in known_gallery_fake_attr_dict:
                # 필터링 방법에 따라 계산
                if FILTERING_METHOD == "top":
                    # 모델이 예측한 상의 하의 색 중 가장 확률이 높은 것
                    gallery_fake_up_set = get_topk_colors(attr=gallery_attr[gallery_img_path], start_idx=18, end_idx=26, k=TOP_UP_K)
                    gallery_fake_down_set = get_topk_colors(attr=gallery_attr[gallery_img_path], start_idx=4, end_idx=13, k=TOP_DOWN_K)
                elif FILTERING_METHOD == "over05":
                    gallery_fake_up_set = get_over_n_colors(attr=gallery_attr[gallery_img_path], start_idx=18, end_idx=26, threshold=THRESHOLD)
                    gallery_fake_down_set = get_over_n_colors(attr=gallery_attr[gallery_img_path], start_idx=4, end_idx=13, threshold=THRESHOLD)                   
                # 필터링 방법에 따라 계산된 
                known_gallery_fake_attr_dict[gallery_img_path][0] = gallery_fake_up_set
                known_gallery_fake_attr_dict[gallery_img_path][1] = gallery_fake_down_set
            else:
                gallery_fake_up_set = known_gallery_fake_attr_dict[gallery_img_path][0]
                gallery_fake_down_set = known_gallery_fake_attr_dict[gallery_img_path][1]

            # print()
            # print(img_path)
            gallery_real_up_idx = gallery_real_attribute[gallery_img_path][0]
            gallery_real_down_idx = gallery_real_attribute[gallery_img_path][1]

            # 만약 갤러리에서 라벨을 판단하기 애매하여 설정하지 않은 경우
            if(gallery_real_up_idx == 0):
                gallery_real_up_list = gallery_fake_up_set
            else:
                gallery_real_up_list = {up_color_dict[gallery_real_up_idx]}
            if(gallery_real_down_idx == 0):
                gallery_real_down_list = gallery_fake_down_set
            else:
                gallery_real_down_list = {down_color_dict[gallery_real_down_idx]}

            # set
            if (set(query_fake_up_set) & set(gallery_fake_up_set)) and (set(query_fake_down_set) & set(gallery_fake_down_set)):
                remain_filtered_gallery_features_dict[gallery_img_path] = gallery_fake_features_attr[gallery_img_path]
                remain_gallery_path_list_per_query_img[query_img_path].append(gallery_img_path)
                if (set(query_fake_up_set) & set(gallery_real_up_list)) and (set(query_fake_down_set) & set(gallery_real_down_list)):
                    tn += 1
                else:
                    fp += 1
            else:
                if (set(query_fake_up_set) & set(gallery_real_up_list)) and (set(query_fake_down_set) & set(gallery_real_down_list)):
                    fn += 1
                else:
                    tp += 1
            # 신원별 남아있는 갤러리 수 확인 안함.
        percision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1_score = 2 * percision * recall / (percision + recall)
        i += 1
        end_time=time.time()
        print(f'resolution_{(iter_num+1)*10}_{os.path.basename(query_img_path)} 쿼리 결과 -','F1-score:', F1_score, f'fake_up:{query_fake_up_set}, fake_down:{query_fake_down_set}, time: {round(end_time-start_time,2)}초. {i}/{len(query_img_path_list)}. ')
        # print(f"신원별 남아있는 이미지 수: {gal_filter_dict}")
        print("True Positive (TP):", tp)
        print("False Positive (FP):", fp)
        print("True Negative (TN):", tn)
        print("False Negative (FN):", fn)

        # count_list.append(count)
        # id_list.append(len(gal_filter_dict))
        TP_list.append(tp)
        FP_list.append(fp)
        TN_list.append(tn)
        FN_list.append(fn)
        F1_list.append(F1_score)
    return TP_list, FP_list, TN_list, FN_list, F1_list, remain_filtered_gallery_features_dict, remain_gallery_path_list_per_query_img
    # cal dist
async def main(args):
    MAX_WORKER, SAVED_COLOR, FILTERING_METHOD, TOP_UP_K, TOP_DOWN_K = args
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=MAX_WORKER) as executor:
        futures = [
            loop.run_in_executor(executor, generate_filtered_gallery, i, FILTERING_METHOD, TOP_UP_K, TOP_DOWN_K)
            for i in range(9, -1, -1)
        ]
        results = await asyncio.gather(*futures)
        for i, result in enumerate(results):
            TP_list, FP_list, TN_list, FN_list, F1_list, remain_filtered_gallery_features_dict, remain_gallery_path_list_per_query_img = result

            remain_gallery_path_list_per_query_img_save_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/filtered_gallery/meta/remain_gallery_path_list_per_query_img_{(i+1) * 10}.pkl"
            save_data_as_pkl(remain_gallery_path_list_per_query_img, remain_gallery_path_list_per_query_img_save_path)

            filtered_gallery_save_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/filtered_gallery/bounding_box_test_{(i+1) * 10}"
            remain_img_path = remain_filtered_gallery_features_dict.keys()
            print(f"filtered gallery: {len(remain_img_path)}")
            for img_path in remain_img_path:
                img_name = os.path.basename(img_path)
                destination_file = filtered_gallery_save_path + "/" + img_name
                shutil.copyfile(img_path, destination_file)

            f1_score_save_path = f"/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false/filtering/f1_score_{(i+1)*10}.txt"
            output_str = f"TP: {sum(TP_list)/len(TP_list)}\n"
            output_str += f"True Positive (TP): {sum(TP_list)/len(TP_list)}\n"
            output_str += f"False Positive (FP): {sum(FP_list)/len(FP_list)}\n"
            output_str += f"True Negative (TN): {sum(TN_list)/len(TN_list)}\n"
            output_str += f"False Negative (FN): {sum(FN_list)/len(FN_list)}\n"
            output_str += f"F1-score: {sum(F1_list)/len(F1_list)}\n"
            with open(f1_score_save_path, 'w') as file:
                file.write(output_str)

if __name__ == '__main__':
    MAX_WORKER = 4
    SAVED_COLOR = False
    FILTERING_METHOD = "top"
    TOP_UP_K = 1
    TOP_DOWN_K = 1
    #FILTERING_METHOD = "over0.5"
    args = (MAX_WORKER, SAVED_COLOR, FILTERING_METHOD, TOP_UP_K, TOP_DOWN_K)
    asyncio.run(main(args))
