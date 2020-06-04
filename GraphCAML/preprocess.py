# get all entityid2index appear in the dataset
import json
import numpy as np
import pandas as pd

user2index = {}
item2index = {}

# file_name = './data/reviews_Musical_Instruments_5.json'
# file_name = './data/reviews_Automotive_5.json'
file_name = './data/reviews_Patio_Lawn_and_Garden_5.json'
# file_name = './data/reviews_Amazon_Instant_Video_5.json'
# file_name = './data/reviews_Baby_5.json'

def filter_dataset():
    all_users, all_items, filtered_data = [], [], []
    print('filter dataset...\n')
    reader = open(file_name, 'r', encoding="utf-8")
    user_review_freq = {}
    for line in reader:
        temp = json.loads(line, strict=False)
        userid = temp["reviewerID"]
        if userid not in user_review_freq:
            user_review_freq[userid] = 1
        else:
            user_review_freq[userid] = user_review_freq[userid] + 1
    reader.close()

    reader1 = open(file_name, 'r', encoding="utf-8")
    for line in reader1:
        temp = json.loads(line, strict=False)
        userid = temp["reviewerID"]
        if userid in user_review_freq and user_review_freq[userid] >= 10:
            filtered_data.append(temp)
            all_users.append(temp["reviewerID"])
            all_items.append(temp["asin"])
    reader1.close()
    all_users = list(set(all_users))
    all_items = list(set(all_items))
    print('Finish filtering')
    return all_users, all_items, filtered_data


def map2index(all_users, all_items):
    for i, user in enumerate(all_users):
        user2index[user] = i
    for j, item in enumerate(all_items):
    	item2index[item] = j
    print('Finish mapping')


def convert_dataset():
    transformed_data = []
    for temp in filtered_data:
        transformed_data.append({
			'user': user2index[temp['reviewerID']],
			'item': item2index[temp['asin']],
			'rating': temp['overall'],
			'review': temp['summary'],
			})

    np.random.shuffle(transformed_data) 
    reviews = pd.DataFrame(transformed_data)
    reviews.to_csv('./data/reviews.csv')
    print('Finish gererating csv file')
    return  transformed_data


def split_dataset(transformed_data):
    print('spliting dataset ...')
    writer_train = open('./data/train.json', 'w', encoding='utf-8')
    writer_valid = open('./data/valid.json', 'w', encoding='utf-8')
    writer_test = open('./data/test.json', 'w', encoding='utf-8')

    # train_dataset:eval:test = 8:1:1
    eval_ratio = 0.1
    test_ratio = 0.1
    all_data = []
    train_data = []
    valid_data = []
    test_data = []
    user_train = []
    item_train = []

    num = len(transformed_data)

    cnt = 0
    num1 = int(num * (1 - eval_ratio - test_ratio))
    num2 = int(num * (1 - test_ratio))

    for temp in transformed_data:
        if cnt < num1:
            train_data.append(temp)
            user_train.append(temp["user"])
            item_train.append(temp["item"])
        elif cnt < num2:
            if temp["user"] in user_train and temp["item"] in item_train :
                valid_data.append(temp)
        else:
            if temp["user"] in user_train and temp["item"] in item_train:
                test_data.append(temp)
        cnt = cnt + 1

    writer_train.write(json.dumps(train_data, ensure_ascii=False))
    writer_valid.write(json.dumps(valid_data, ensure_ascii=False))
    writer_test.write(json.dumps(test_data, ensure_ascii=False))
    print('Finish splitting')


print("start preprocessing dataset", file_name)
all_users, all_items , filtered_data = filter_dataset() 
map2index(all_users, all_items)
transformed_data = convert_dataset()
split_dataset(transformed_data)
# print(len(list(user2index.items())))
# print(len(list(item2index.items())))