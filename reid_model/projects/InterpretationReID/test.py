import pandas as pd

# 예시 데이터프레임 생성
data = {
    'clothes': [7, 8, 9],
    'downblack': [13, 14, 15],
    'backpack': [1, 2, 3],
    'bag': [4, 5, 6],

    'down': [10, 11, 12],

    'downblue': [16, 17, 18],
    'downbrown': [19, 20, 21],
    'downgray': [22, 23, 24],
    'downgreen': [25, 26, 27],
    'downpink': [28, 29, 30],
    'downpurple': [31, 32, 33],
    'downwhite': [34, 35, 36],
    'downyellow': [37, 38, 39],
    'gender': [40, 41, 42],
    'hair': [43, 44, 45],
    'handbag': [46, 47, 48],
    'hat': [49, 50, 51],
    'up': [52, 53, 54],
    'upblack': [55, 56, 57],
    'upblue': [58, 59, 60],
    'upgray': [61, 62, 63],
    'upgreen': [64, 65, 66],
    'uppurple': [67, 68, 69],
    'upred': [70, 71, 72],
    'upwhite': [73, 74, 75],
    'upyellow': [76, 77, 78]
}

# 데이터프레임 생성
df = pd.DataFrame(data)


origin_label_list = ['backpack', 'bag', 'clothes', 'down', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'gender', 'hair', 'handbag', 'hat', 'up', 'upblack', 'upblue', 'upgray', 'upgreen', 'uppurple', 'upred', 'upwhite', 'upyellow']

# 인덱스 재정렬
df = df[origin_label_list].T.to_dict('list')
print(df)
#dict = df.T.to_dict('list')

# 재정렬된 데이터프레임 출력
#print(df)
