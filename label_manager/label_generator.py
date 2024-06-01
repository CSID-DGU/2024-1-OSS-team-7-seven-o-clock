import pandas as pd
import os
import shutil

# Load the correctly uploaded Excel file
root_path = os.getcwd()
# label_manager_path = os.path.join(root_path, "label_statistics")
label_manager_path = root_path
dataset_dir_path = os.path.join(label_manager_path, "dataset")
dataset_path = os.path.join(dataset_dir_path, "Market-1501-v15.09.15")
train_data_path = os.path.join(dataset_path, "bounding_box_train")
test_data_path = os.path.join(dataset_path, "bounding_box_test")
query_data_path = os.path.join(dataset_path, "query")
correct_label_path = os.path.join(label_manager_path, "correct_label.xlsx")
new_dataset_path = os.path.join(dataset_dir_path, "Market-1501-v24.05.20")
new_train_data_path = os.path.join(new_dataset_path, "bounding_box_train")
new_test_data_path = os.path.join(new_dataset_path, "bounding_box_test")
new_query_data_path = os.path.join(new_dataset_path, "query")
csv_result_dir_path = os.path.join(label_manager_path, "csv")


def print_columns_with_nan(df):
    columns_with_nan = df.columns[df.isna().any()].tolist()
    print(f"Columns with NaN values: {columns_with_nan}")
    return columns_with_nan

def save_statistical_xlsx(df, output_file_path):
    # Drop the 'new_img_name' column
    df = df.drop(columns=['new_img_name'])

    # Save the DataFrame to an xlsx file
    df.to_excel(output_file_path, index=False)

    print(f"DataFrame saved to {output_file_path}")

def get_filtered_file_list(**directory_paths):
    file_lists = {}
    for key, directory_path in directory_paths.items():
        file_list = []
        if os.path.exists(directory_path):
            for file_name in os.listdir(directory_path):
                if not (file_name.startswith('0000') or file_name.startswith('-1')) and file_name.endswith('.jpg'):
                    file_name = file_name.replace('.jpg.jpg', '.jpg')
                    file_list.append(file_name)
        file_lists[key] = file_list
    return file_lists


def update_new_img_name(df, pid_range = 1587, junk_option = False):
    origin_pid_set = set(range(1, pid_range+1))
    def transform_name(row):
        origin_img_name = row['origin_img_name']
        new_img_name = row['new_img_name']
        origin_id, rest = origin_img_name.split('_', 1)
        if int(origin_id) in origin_pid_set:
            origin_pid_set.discard(int(origin_id))
        
        if new_img_name == 0:
            return origin_img_name
        elif new_img_name == -1 or junk_option:
            return f"-1_{rest}"
        # junk option == true 라면 데이터셋 자체가 변화하지 않는다.
        else:        
            return f"{new_img_name:04d}_{rest}"
        
    def transform_junk(row):
        origin_img_name = row['origin_img_name']
        new_img_name = row['new_img_name']
        origin_id, rest = origin_img_name.split('_', 1)
        if int(origin_id) in origin_pid_set:
            origin_pid_set.discard(int(origin_id))
        
        # 0인 경우에는 무조건 그대로
        if new_img_name == 0:
            return origin_img_name
        # junk_option == True거나 번호가 -1인 경우는 무조건 -1로
        elif junk_option or new_img_name == -1:
            return f"-1_{rest}"
        # junk_option == False 이고 0도, -1도 아니고 1585 같은 숫자가 써있다면 무시하고 그냥 원래 이름으로.
        else:
            return origin_img_name    
        
    if pid_range == 1587:
        df['new_img_name'] = df.apply(transform_name, axis=1)
        print("not find: ", origin_pid_set)
    elif pid_range == 1501:
        df['new_img_name'] = df.apply(transform_junk, axis=1)
        print("not find: ", origin_pid_set)
    return df

def load_xlsx_and_init(file_path):
    df = pd.read_excel(file_path)
    # Step 1: Delete rows where the origin_img_name length is less than 18
    df = df[df['origin_img_name'].apply(len) >= 18]
    df['new_img_name'].fillna(0, inplace=True)
    df = df.drop_duplicates(subset=['origin_img_name'])

    print_columns_with_nan(df)

    # Step 2: Ensure .jpg is appended to values in the origin_img_name column if not already present
    df['origin_img_name'] = df['origin_img_name'].apply(lambda x: x if x.endswith('.jpg') else f"{x}.jpg")
    df['origin_img_name'] = df['origin_img_name'].apply(lambda x: x.replace('.jpg.jpg', '.jpg'))

    # Step 3: Remove duplicates from the origin_img_name column
    df = df.drop_duplicates(subset=['origin_img_name'])
    columns_to_convert = df.columns.difference(['origin_img_name'])
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    df = df.sort_values(by='origin_img_name')

    # Display the updated DataFrame
    print(df)
    return df

def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")



# Function to categorize image names
def categorize_img_name(img_name, train_data_set, test_data_set, query_data_set):
    if img_name in train_data_set:
        return 'train'
    elif img_name in test_data_set:
        return 'test'
    elif img_name in query_data_set:
        return 'query'
    else:
        return 'unknown'

def copy_and_rename_files(df):
    os.makedirs(new_train_data_path, exist_ok=True)
    os.makedirs(new_test_data_path, exist_ok=True)
    os.makedirs(new_query_data_path, exist_ok=True)

    for index, row in df.iterrows():
        origin_img_name = row['origin_img_name']
        new_img_name = row['new_img_name']
        dataset_category = row['dataset_category']
        
        if dataset_category == 'train':
            source_path = os.path.join(train_data_path, origin_img_name)
            dest_path = os.path.join(new_train_data_path, new_img_name)
        elif dataset_category == 'test':
            source_path = os.path.join(test_data_path, origin_img_name)
            dest_path = os.path.join(new_test_data_path, new_img_name)
        elif dataset_category == 'query':
            source_path = os.path.join(query_data_path, origin_img_name)
            dest_path = os.path.join(new_query_data_path, new_img_name)
        else:
            continue
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy and rename the file
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
        else:
            print(f"File {source_path} does not exist")

def get_special_file_list(directory_path):
    '''
        0000 또는 -1로 시작하는 사진 이름 리스트 반환 
    '''
    file_list = []
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            if (file_name.startswith('0000') or file_name.startswith('-1')) and file_name.endswith('.jpg'):
                file_list.append(file_name)
    return file_list

def copy_special_files(src_path, dest_path):
    '''
     위 함수를 이용해서 0000, -1로 시작하는 파일들을 복사
    '''
    special_files = get_special_file_list(src_path)
    for file_name in special_files:
        source_path = os.path.join(src_path, file_name)
        destination_path = os.path.join(dest_path, file_name)
        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
        else:
            print(f"Special file {source_path} does not exist")

if __name__ == '__main__':
    df = load_xlsx_and_init(correct_label_path)
    # save_statistical_xlsx(df, root_path+"/for_statistics.xlsx")
    file_lists = get_filtered_file_list(train=train_data_path, test=test_data_path, query=query_data_path)
    origin_img_name_list = file_lists['train'] + file_lists['test'] + file_lists['query']
    origin_img_name_list.sort()


    img_name_list = df['origin_img_name'].tolist()
    img_name_list.sort()

    missing_in_img_name_list = [img for img in origin_img_name_list if img not in img_name_list]
    if len(missing_in_img_name_list) > 0:
        print(missing_in_img_name_list)
        exit()
    # 1501 옵션을 넣으면 데이터셋 자체는 변화시키지 않고, 1501명으로 유지하지만 라벨만 바꿈
    # junk 옵션을 넣으면 pid가 변해야 하는 사진을 모두 junk로 처리함.
    df = update_new_img_name(df, pid_range=1501, junk_option=False)
    
    train_data_set = set(file_lists['train'])
    test_data_set = set(file_lists['test'])
    query_data_set = set(file_lists['query'])
    df['dataset_category'] = df['origin_img_name'].apply(lambda img_name: categorize_img_name(img_name, train_data_set, test_data_set, query_data_set))
    save_to_csv(df, csv_result_dir_path + "/correct_label_intermid.csv")

    copy_and_rename_files(df)

    copy_special_files(test_data_path, new_test_data_path)

    df = df.drop(columns=['origin_img_name'])
    df = df.rename(columns={'new_img_name': 'img_name'})

    # 'img_name' 열의 값을 '_' 기준으로 분할하고 첫 번째 값을 추출
    df['pid'] = df['img_name'].str.split('_').str[0]

    # 첫 번째 값이 '-1'인 행을 필터링하여 제거
    df = df[df['pid'] != '-1']

    # 'first_part' 열을 삭제 (필요 시)
    df.drop(columns=['pid'], inplace=True)

    # 기존 연구와 동일한 순서로 속성 이름을 변경
    columns = [ 'img_name', 'age', 'backpack', 'bag', 'clothes', 'down', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'gender', 'hair', 'handbag', 'hat', 'up', 'upblack', 'upblue', 'upgray', 'upgreen', 'uppurple', 'upred', 'upwhite', 'upyellow', 'dataset_category']
    df = df[columns]

    # Split the DataFrame into train, test, and query DataFrames
    df_train = df[df['dataset_category'] == 'train'].drop(columns='dataset_category')
    df_test = df[df['dataset_category'] == 'test'].drop(columns='dataset_category')
    df_query = df[df['dataset_category'] == 'query'].drop(columns='dataset_category')

    # Save the updated DataFrames to CSV
    save_to_csv(df_train, csv_result_dir_path + "/train_attribute.csv")
    save_to_csv(df_test, csv_result_dir_path + "/test_attribute.csv")
    save_to_csv(df_query, csv_result_dir_path + "/query_attribute.csv")


    
