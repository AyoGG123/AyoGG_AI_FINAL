# 处理lan8数据
import os
import re
import sys
import json
import random
import pandas as pd
from argparse import ArgumentParser
from opencc import OpenCC
from os import listdir
# from facebook_mbart_model import *
from GPT import *
from datasets import load_dataset
from sklearn.utils import resample

cc = OpenCC('tw2s')
ROOT = os.getcwd()
print(ROOT)


def trans_generated_json_bart():
    # fortune_telling        list
    # Questions to Ask       list
    # keywords               list
    # meanings               dict list
    # Mythical / Spiritual   str
    data = read_json()

    for i in data['cards']:
        process = []
        for line in i['fortune_telling']:
            process.append(line)
        for line in i['Questions to Ask']:
            process.append(line)
        for line in i['keywords']:
            process.append(line)
        for key in i['meanings']:
            temp = i['meanings']
            for line in temp[key]:
                process.append(line)

        After_Processing = model_eval(process)

        index = 0

        for l, line in enumerate(i['fortune_telling']):
            i['fortune_telling'][l] = After_Processing[index]
            index += 1
        for l, line in enumerate(i['Questions to Ask']):
            i['Questions to Ask'][l] = After_Processing[index]
            index += 1
        for l, line in enumerate(i['keywords']):
            i['keywords'][l] = After_Processing[index]
            index += 1
        for key in i['meanings']:
            temp = i['meanings']
            for l, line in enumerate(temp[key]):
                temp[key][l] = After_Processing[index]
                index += 1

    # 指定要保存的文件名
    file_name = 'output.json'

    # 打開文件並保存字典內容
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def trans_generated_json_gpt():
    # fortune_telling        list
    # Questions to Ask       list
    # keywords               list
    # meanings               dict list
    # Mythical / Spiritual   str

    input_token = 0
    output_token = 0

    data = read_json()

    for i in data['cards']:
        process = []
        for line in i['fortune_telling']:
            process.append(line)
        for line in i['Questions to Ask']:
            process.append(line)
        for line in i['keywords']:
            process.append(line)
        for key in i['meanings']:
            temp = i['meanings']
            for line in temp[key]:
                process.append(line)

        After_Processing, a, b = GPT_response_list(process)
        input_token += a
        output_token += b

        index = 0

        for l, line in enumerate(i['fortune_telling']):
            i['fortune_telling'][l] = After_Processing[index]
            index += 1
        for l, line in enumerate(i['Questions to Ask']):
            i['Questions to Ask'][l] = After_Processing[index]
            index += 1
        for l, line in enumerate(i['keywords']):
            i['keywords'][l] = After_Processing[index]
            index += 1
        for key in i['meanings']:
            temp = i['meanings']
            for l, line in enumerate(temp[key]):
                temp[key][l] = After_Processing[index]
                index += 1

    # 指定要保存的文件名
    file_name = 'output.json'
    print(f"input_token：{input_token}")
    print(f"output_token ：{output_token}")
    # 打開文件並保存字典內容
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def trans_generated_csv_bart():
    df = read_csv()  # Index(['Card 1', 'Card 2', 'Card 3', 'Reading'], dtype='object')
    process = list(df['Reading'])
    After_Processing = model_eval(process)
    df['Reading'] = After_Processing  # 替換掉 df['Reading']
    output_file_path = 'output.csv'
    df.to_csv(output_file_path, index=False, encoding='utf-8')


def trans_generated_csv_gpt():
    input_token = 0
    output_token = 0
    df = read_csv()  # Index(['Card 1', 'Card 2', 'Card 3', 'Reading'], dtype='object')
    process = list(df['Reading'])
    After_Processing, a, b = GPT_response_list(process)
    input_token += a
    output_token += b
    df['Reading'] = After_Processing  # 替換掉 df['Reading']
    print(f"input_token：{input_token}")
    print(f"output_token ：{output_token}")
    output_file_path = 'output.csv'
    df.to_csv(output_file_path, index=False, encoding='utf-8')


def read_json(filename=os.path.join(ROOT, '塔羅', 'Tarot Deck', 'tarot-images.json')):
    # 文件路徑
    # filename = os.path.join(ROOT, '塔羅', 'Tarot Deck', 'tarot-images.json')
    # filename = os.path.join(ROOT, '塔羅', 'Tarot Deck', 'GPT4o.json')
    # filename = os.path.join(ROOT, '塔羅', 'Tarot Deck', 'mbart.json')

    # 打開並讀取 JSON 文件
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def read_csv(filename=os.path.join(ROOT, '塔羅', 'mbart.csv')):
    # 文件路徑

    # filename = os.path.join(ROOT, '塔羅', 'tarot_readings.csv')
    # filename = os.path.join(ROOT, '塔羅', 'mbart.csv')
    # df = load_dataset(path='csv',
    #                   data_files=filename)
    df = pd.read_csv(filename)

    return df


def rename(input_str):
    # 分割字串
    str_list = input_str.split(' ')
    # 將第一個單詞首字母大寫
    str_list[0] = str_list[0].capitalize()
    # 將其餘單詞轉換為小寫
    for i in range(1, len(str_list)):
        str_list[i] = str_list[i].lower()
    # 合併字串
    result_str = ' '.join(str_list)
    return result_str


def Name_Regularization_json():
    data = read_json()
    for i in data['cards']:
        i['name'] = rename(i['name'])
    # 指定要保存的文件名
    file_name = 'output.json'
    # 打開文件並保存字典內容
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def Name_Regularization_csv():
    df = read_csv()  # Index(['Card 1', ' Card 2', ' Card 3', ' Reading'], dtype='object')

    for col in ['Card 1', 'Card 2', 'Card 3']:
        df[col] = df[col].apply(rename)

    output_file_path = 'output.csv'
    df.to_csv(output_file_path, index=False, encoding='utf-8')


def filter_long_words(text, min_length=5):
    # 使用正則表達式找到所有的英文字
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    # 篩選出長度超過 min_length 的字
    # long_words = [word for word in words if len(word) > min_length]
    if len(words) > min_length:
        return True

    return False


def max_length_reading():
    df = read_csv(filename=os.path.join(ROOT, '塔羅',
                                        'mbart.csv'))  # Index(['Card 1', 'Card 2', 'Card 3', 'Reading'], dtype='object')
    process = list(df['Reading'])
    Poor_translation = []
    max_length = 0
    max_length_str = ""
    for i, line in enumerate(process):
        if filter_long_words(line):
            Poor_translation.append(i)
            print(f"{line}")
        if len(line) > max_length:
            max_length = len(line)
            max_length_str = line
    print(f"最大長度為{max_length}\n"
          f"{max_length_str}")
    print(Poor_translation)


def GPT_remedy(SOURCE, TARGET, BAD_LIST):
    df_source = pd.read_csv(SOURCE)
    df_target = pd.read_csv(TARGET)

    process = list(df_target['Reading'])

    for i, line in enumerate(df_source['Reading']):
        if i in BAD_LIST:
            process[i] = GPT_response_sentence(line)
            pass

    df_target['Reading'] = process  # 替換掉 df['Reading']

    output_file_path = 'output.csv'
    df_target.to_csv(output_file_path, index=False, encoding='utf-8')


def tarot_name_json():
    data = read_json(os.path.join(ROOT, '塔羅', 'Tarot Deck', 'mbart.json'))
    name = []
    for i in data['cards']:
        name.append(i['name'])
    name.sort()
    print(name)


def tarot_name_csv():
    df = read_csv(filename=os.path.join(ROOT, '塔羅', 'mbart.csv'))
    name = set()
    for a, b, c in zip(df['Card 1'], df['Card 2'], df['Card 3']):
        name.add(a)
        name.add(b)
        name.add(c)
    name = list(name)
    name.sort()
    print(name)


def simple_chinese_csv():
    df = read_csv(filename=os.path.join(ROOT, '塔羅',
                                        'mbart.csv'))  # Index(['Card 1', 'Card 2', 'Card 3', 'Reading'], dtype='object')

    process = list(df['Reading'])
    process = [cc.convert(i) for i in process]
    df['Reading'] = process  # 替換掉 df['Reading']

    output_file_path = 'output.csv'
    df.to_csv(output_file_path, index=False, encoding='utf-8')


def simple_chinese_json():
    # fortune_telling        list
    # Questions to Ask       list
    # keywords               list
    # meanings               dict list
    # Mythical / Spiritual   str
    data = read_json(os.path.join(ROOT, '塔羅', 'Tarot Deck', 'GPT4o.json'))

    for i in data['cards']:
        for l, line in enumerate(i['fortune_telling']):
            i['fortune_telling'][l] = cc.convert(line)
        for l, line in enumerate(i['Questions to Ask']):
            i['Questions to Ask'][l] = cc.convert(line)
        for l, line in enumerate(i['keywords']):
            i['keywords'][l] = cc.convert(line)
        for key in i['meanings']:
            temp = i['meanings']
            for l, line in enumerate(temp[key]):
                temp[key][l] = cc.convert(line)

    # 指定要保存的文件名
    file_name = 'output.json'

    # 打開文件並保存字典內容
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


BAD = [43, 81, 100, 144, 194, 199, 266, 286, 321, 378, 540, 546, 647, 689, 711, 738, 884, 941, 1012, 1119, 1154, 1191,
       1212,
       1281, 1451, 1486, 1491, 1598, 1608, 1613, 1657, 1672, 1683, 1867, 1932, 2053, 2168, 2189, 2215, 2356, 2416, 2454,
       2492,
       2531, 2547, 2576, 2611, 2677, 2749, 2772, 2880, 2930, 3087, 3311, 3370, 3481, 3487, 3488, 3615, 3666, 3726, 3849,
       3899,
       4035, 4047, 4360, 4467, 4507, 4511, 4581, 4640, 4648, 4690, 4756, 4765, 4995, 5030, 5033, 5057, 5105, 5128, 5225,
       5281,
       5313, 5321, 5327, 5331, 5399, 5442, 5458, 5561, 5574, 5575]

if __name__ == "__main__":
    # read_json()
    # trans_generated_json_bart()
    # trans_generated_json_gpt()
    # Name_Regularization_json()
    # Name_Regularization_csv()
    # trans_generated_csv_bart()
    # max_length_reading()
    # GPT_remedy(SOURCE=os.path.join(ROOT, '塔羅', 'tarot_readings.csv'),
    #            TARGET=os.path.join(ROOT, '塔羅', 'mbart.csv'),
    #            BAD_LIST=BAD)
    # tarot_name_json()
    # tarot_name_csv()
    # simple_chinese_csv()
    # simple_chinese_json()
    print(cc.convert(
        f"我抽出了，，，請詳細解釋這個牌陣的涵義，包括每張牌在不同位置的象徵意義，以及它們之間的相互影響和整體解讀。希望能透過這個牌陣，了解當前情境、挑戰及未來可能的發展方向。"))
