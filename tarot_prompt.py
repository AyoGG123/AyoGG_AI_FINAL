import os
import re
import sys
import json
import random
import pandas as pd
from opencc import OpenCC
from os import listdir
from datasets import load_dataset
from sklearn.utils import resample

cc = OpenCC('tw2s')
ROOT = os.getcwd()
print(ROOT)


def get_data(file):
    df = pd.read_csv(file)
    dataList = []
    labelList = []
    for i, o in zip(df['input'], df['output']):
        dataList.append(i)
        labelList.append(o)

    return dataList, labelList


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


def json_Classification(data):
    # min 1 3 4 3 2
    return_dict = {"fortune_telling": data["fortune_telling"],
                   "keywords": data["keywords"],
                   "light": data["meanings"]["light"],
                   "shadow": data["meanings"]["shadow"],
                   "Questions to Ask": data["Questions to Ask"]}
    return return_dict


def pick_card(data, name):
    for card in data["cards"]:
        if card["name"] == name:
            return card
    print('bug')


cards = ['Ace of cups', 'Ace of pentacles', 'Ace of swords', 'Ace of wands', 'Death', 'Eight of cups',
         'Eight of pentacles', 'Eight of swords', 'Eight of wands', 'Five of cups', 'Five of pentacles',
         'Five of swords', 'Five of wands', 'Four of cups', 'Four of pentacles', 'Four of swords', 'Four of wands',
         'Judgement', 'Justice', 'King of cups', 'King of pentacles', 'King of swords', 'King of wands',
         'Knight of cups', 'Knight of pentacles', 'Knight of swords', 'Knight of wands', 'Nine of cups',
         'Nine of pentacles', 'Nine of swords', 'Nine of wands', 'Page of cups', 'Page of pentacles',
         'Page of swords', 'Page of wands', 'Queen of cups', 'Queen of pentacles', 'Queen of swords',
         'Queen of wands', 'Seven of cups', 'Seven of pentacles', 'Seven of swords', 'Seven of wands', 'Six of cups',
         'Six of pentacles', 'Six of swords', 'Six of wands', 'Strength', 'Temperance', 'Ten of cups',
         'Ten of pentacles', 'Ten of swords', 'Ten of wands', 'The chariot', 'The devil', 'The emperor',
         'The empress', 'The fool', 'The hanged man', 'The hermit', 'The hierophant', 'The high priestess',
         'The lovers', 'The magician', 'The moon', 'The star', 'The sun', 'The tower', 'The world', 'Three of cups',
         'Three of pentacles', 'Three of swords', 'Three of wands', 'Two of cups', 'Two of pentacles',
         'Two of swords', 'Two of wands', 'Wheel of fortune']


def tarot_prompt_generated(Card_1, Card_2, Card_3, Reading, json_file=os.path.join(ROOT, '塔羅', 'Tarot Deck',
                                                                                   'GPT4o.json')):  # Card_1-3 塔羅卡名 str ,Reading
    data = read_json(json_file)

    Card_1_dict = json_Classification(pick_card(data=data, name=Card_1))
    Card_2_dict = json_Classification(pick_card(data=data, name=Card_2))
    Card_3_dict = json_Classification(pick_card(data=data, name=Card_3))

    preface = ["首先，让我们先分析每张牌的象征意义。", "首先，让我们先对每张牌进行解释。",
               "让我们从每张牌的象征意义开始分析。", "首先，我们来解释每张牌的含义。",
               "让我们先来剖析每张牌背后的意义。", "首先，我们将逐一解析每张牌的象征。",
               "让我们从解释每张牌的意义开始。", "让我们先探讨每张牌的象征意涵。", "首先，我们来分析每张牌的意义。",
               "让我们先理解每张牌的含义。", "首先，我们来解析每张牌的象征。", "让我们从每张牌的解释开始。",
               "让我们先来探讨每张牌的意义。", "首先，我们来研究每张牌的象征。", "让我们先分析每张牌的含义。",
               "让我们先分析每张牌的含义。", "首先，我们来剖析每张牌的意涵。", "让我们从解释每张牌的象征开始。"]

    preface = random.sample(preface, 1)[0]

    keywords = random.sample(Card_1_dict["keywords"], 2)

    feature1 = [f"{Card_1}:代表了{keywords[0]}和{keywords[1]}。", f"{Card_1}:象征着{keywords[0]}和{keywords[1]}。",
                f"{Card_1}:意味着{keywords[0]}和{keywords[1]}。", f"{Card_1}:代表着{keywords[0]}和{keywords[1]}。",
                f"{Card_1}:意指{keywords[0]}和{keywords[1]}。", f"{Card_1}:的象征是{keywords[0]}和{keywords[1]}。",
                f"{Card_1}:表现了{keywords[0]}与{keywords[1]}。", f"{Card_1}:意味着{keywords[0]}、{keywords[1]}的结合。",
                f"{Card_1}:意指{keywords[0]}与{keywords[1]}的融合。",
                f"{Card_1}:的意涵包含了{keywords[0]}和{keywords[1]}。",
                f"{Card_1}:的特质包含了{keywords[0]}和{keywords[1]}。"]
    feature1 = random.sample(feature1, 1)[0]

    keywords = random.sample(Card_2_dict["keywords"], 2)
    feature2 = [f"{Card_2}:代表了{keywords[0]}和{keywords[1]}。", f"{Card_2}:象征着{keywords[0]}和{keywords[1]}。",
                f"{Card_2}:意味着{keywords[0]}和{keywords[1]}。", f"{Card_2}:代表着{keywords[0]}和{keywords[1]}。",
                f"{Card_2}:意指{keywords[0]}和{keywords[1]}。", f"{Card_2}:的象征是{keywords[0]}和{keywords[1]}。",
                f"{Card_2}:表现了{keywords[0]}与{keywords[1]}。", f"{Card_2}:意味着{keywords[0]}、{keywords[1]}的结合。",
                f"{Card_2}:意指{keywords[0]}与{keywords[1]}的融合。",
                f"{Card_2}:的意涵包含了{keywords[0]}和{keywords[1]}。",
                f"{Card_2}:的特质包含了{keywords[0]}和{keywords[1]}。"]
    feature2 = random.sample(feature2, 1)[0]

    keywords = random.sample(Card_3_dict["keywords"], 2)
    feature3 = [f"{Card_3}:代表了{keywords[0]}和{keywords[1]}。", f"{Card_3}:象征着{keywords[0]}和{keywords[1]}。",
                f"{Card_3}:意味着{keywords[0]}和{keywords[1]}。", f"{Card_3}:代表着{keywords[0]}和{keywords[1]}。",
                f"{Card_3}:意指{keywords[0]}和{keywords[1]}。", f"{Card_3}:的象征是{keywords[0]}和{keywords[1]}。",
                f"{Card_3}:表现了{keywords[0]}与{keywords[1]}。", f"{Card_3}:意味着{keywords[0]}、{keywords[1]}的结合。",
                f"{Card_3}:意指{keywords[0]}与{keywords[1]}的融合。",
                f"{Card_3}:的意涵包含了{keywords[0]}和{keywords[1]}。",
                f"{Card_3}:的特质包含了{keywords[0]}和{keywords[1]}。"]
    feature3 = random.sample(feature3, 1)[0]

    fortune_telling = random.sample(Card_1_dict["fortune_telling"], 1)[0]
    light = random.sample(Card_1_dict["light"], 4)
    shadow = random.sample(Card_1_dict["shadow"], 3)

    past_explain = [f"{Card_1}代表着{light[0]}。它暗示你已经{shadow[0]}。",
                    f"{Card_1}代表{light[0]}，这可能代表你可能经历过了{shadow[0]}。",
                    f"在这个位置，{Card_1}意味着你过去已经{light[0]}。它可能暗示{shadow[0]}。",
                    f"{Card_1}象征着{light[0]}。它暗示你可能面临{shadow[0]}。",
                    f"{Card_1}在牌阵中代表着{light[0]}。它暗示你可能{shadow[0]}，或者是{shadow[1]}。",
                    f"{Card_1}在此位置代表{light[0]}。它暗示你可能面临{shadow[0]}。",
                    f"当前的{Card_1}意味着{light[0]}。这张牌暗示你可能{shadow[0]}。",
                    f"{Card_1}代表{light[0]}。它可能暗示{shadow[0]}。"]
    past_explain = random.sample(past_explain, 1)[0]

    fortune_telling = random.sample(Card_2_dict["fortune_telling"], 1)[0]
    light = random.sample(Card_2_dict["light"], 4)
    shadow = random.sample(Card_2_dict["shadow"], 3)
    present_explain = [f"{Card_2}代表着{light[0]}。它暗示你可能正经历{shadow[0]}。{Card_2}也可以代表{fortune_telling}，",
                       f"这张牌暗示你可能正{shadow[0]}，但也可能正{light[0]}。{Card_2}也象征着{fortune_telling}。",
                       f"{Card_2}在牌阵中代表着{light[0]}。它暗示你可能{shadow[0]}，或者即将{light[1]}。{Card_2}也意味着{fortune_telling}。",
                       f"{Card_2}位于牌阵的中央，代表着{light[0]}。它暗示你可能会{shadow[0]}，这意味着{fortune_telling}。",
                       f"当前的{Card_2}代表{light[0]}，这可能预示着{shadow[0]}。它提醒你要注意{fortune_telling}。",
                       f"在这个位置，{Card_2}意味着{light[0]}。它可能暗示{shadow[0]}，并且预示着{fortune_telling}。",
                       f"这张牌{Card_2}象征着{light[0]}。它暗示你可能面临{shadow[0]}，而这也代表{fortune_telling}。",
                       f"{Card_2}在牌阵中代表着{light[0]}。它暗示你可能{shadow[0]}，或者是{shadow[1]}。{Card_2}也显示出你的预兆是:{fortune_telling}。",
                       f"{Card_2}在此位置代表{light[0]}。它暗示你可能面临{shadow[0]}，而这也象征着{fortune_telling}。",
                       f"当前的{Card_2}意味着{light[0]}。这张牌暗示你可能{shadow[0]}，并且预示着{fortune_telling}。",
                       f"{Card_2}代表{light[0]}。它可能暗示{shadow[0]}，并且也意味着{fortune_telling}。"]
    present_explain = random.sample(present_explain, 1)[0]

    fortune_telling = random.sample(Card_3_dict["fortune_telling"], 1)[0]
    light = random.sample(Card_3_dict["light"], 4)
    shadow = random.sample(Card_3_dict["shadow"], 3)
    feuture_explain = [f"{Card_3}在牌阵中代表着{light[0]}。它暗示你可能{shadow[0]}。{Card_3}也可以代表{fortune_telling}，",
                       f"这张牌暗示你可能正{shadow[0]}，但也可能正{light[0]}。{Card_3}也象征着{fortune_telling}。",
                       f"{Card_3}在牌阵中代表着{light[0]}。它暗示你可能{shadow[0]}，或者即将{light[1]}。{Card_3}也意味着{fortune_telling}。",
                       f"当前的{Card_3}代表{light[0]}，这可能预示着{shadow[0]}。它提醒你要注意{fortune_telling}。",
                       f"在这个位置，{Card_3}意味着{light[0]}。它可能暗示{shadow[0]}，并且预示着{fortune_telling}。",
                       f"这张牌{Card_3}象征着{light[0]}。它暗示你可能面临{shadow[0]}，而这也代表{fortune_telling}。",
                       f"{Card_3}在牌阵中代表着{light[0]}。它暗示你可能{shadow[0]}，或者是{shadow[1]}。{Card_3}也显示出你的预兆是:{fortune_telling}。",
                       f"这张牌位于牌阵的右侧，象征着{light[0]}。它暗示你可能会{shadow[0]}，并且这也意味着{fortune_telling}。",
                       f"{Card_3}在此位置代表{light[0]}。它暗示你可能面临{shadow[0]}，而这也象征着{fortune_telling}。",
                       f"当前的{Card_3}意味着{light[0]}。这张牌暗示你可能{shadow[0]}，并且预示着{fortune_telling}。",
                       f"位于牌阵的上方，{Card_3}代表{light[0]}。它可能暗示{shadow[0]}，并且也意味着{fortune_telling}。"]
    feuture_explain = random.sample(feuture_explain, 1)[0]

    final_part = [f"整体来看：", f"整体而言：", f"最后：", f"整体解读：", f"总体而言："]
    final_part = random.sample(final_part, 1)[0]

    output = f"{preface}\n\n{feature1}{past_explain}\n\n{feature2}{present_explain}\n\n{feature3}{feuture_explain}\n\n{final_part}{Reading}"
    return output


def train_data_generated(csv_file, json_file, output_file):
    df = pd.read_csv(csv_file)
    data_csv = {'input': [], 'output': []}
    response = ""
    instruction = ""

    for c1, c2, c3, read in zip(df['Card 1'], df['Card 2'], df['Card 3'], df['Reading']):
        response = tarot_prompt_generated(Card_1=c1, Card_2=c2, Card_3=c3, Reading=read, json_file=json_file)
        # instruction = f"我抽出了{c1}，{c2}，{c3}，请详细解释这个牌阵的涵义，包括每张牌在不同位置的象征意义，以及它们之间的相互影响和整体解读。希望能透过这个牌阵，了解当前情境、挑战及未来可能的发展方向。"
        instruction = f"我抽出了{c1}，{c2}，{c3}，请详细解释这个牌阵的所代表的過去、現在及未來，包括每张牌在不同位置的象征意义，以及它们之间的相互影响和整体解读。希望能透过这个牌阵，了解当前情境、挑战及未来可能的发展方向。"
        data_csv['input'].append(instruction)
        data_csv['output'].append(response)

    df = pd.DataFrame(data_csv)
    # 將 DataFrame 儲存為 CSV 檔案
    df.to_csv(output_file, index=False, encoding='utf-8-sig')


# explain1 = [f"{Card_1}在牌陣中代表著{light[0]}。它暗示你可能{shadow[0]}。{Card_1}也可以代表{fortune_telling}，",
#                 f"這張牌暗示你可能正{shadow[0]}，但也可能正{light[0]}。{Card_1}也象徵著{fortune_telling}。",
#                 f"{Card_1}在牌陣中代表著{light[0]}。它暗示你可能{shadow[0]}，或者即將{light[1]}。{Card_1}也意味著{fortune_telling}。",
#                 f"這張牌位於牌陣的中央，代表著{light[0]}。它暗示你可能會{shadow[0]}，這意味著{fortune_telling}。",
#                 f"當前的{Card_1}代表{light[0]}，這可能預示著{shadow[0]}。它提醒你要注意{fortune_telling}。",
#                 f"在這個位置，{Card_1}意味著{light[0]}。它可能暗示{shadow[0]}，並且預示著{fortune_telling}。",
#                 f"這張牌{Card_1}象徵著{light[0]}。它暗示你可能面臨{shadow[0]}，而這也代表{fortune_telling}。",
#                 f"位於牌陣的左側，{Card_1}暗示{light[0]}。這意味著{shadow[0]}，並且象徵著{fortune_telling}。",
#                 f"{Card_1}在牌陣中代表著{light[0]}。它暗示你可能{shadow[0]}，或者是{shadow[1]}。{Card_1}也顯示出你的預兆是:{fortune_telling}。",
#                 f"這張牌位於牌陣的右側，象徵著{light[0]}。它暗示你可能會{shadow[0]}，並且這也意味著{fortune_telling}。",
#                 f"{Card_1}在此位置代表{light[0]}。它暗示你可能面臨{shadow[0]}，而這也象徵著{fortune_telling}。",
#                 f"當前的{Card_1}意味著{light[0]}。這張牌暗示你可能{shadow[0]}，並且預示著{fortune_telling}。",
#                 f"位於牌陣的上方，{Card_1}代表{light[0]}。它可能暗示{shadow[0]}，並且也意味著{fortune_telling}。"]

if __name__ == "__main__":
    # tarot_prompt_generated(Card_1=0, Card_2=0, Card_3=0, Reading=0)
    data = read_json(os.path.join(ROOT, '塔羅', 'Tarot Deck', 'GPT4o.json'))
    # min_fortune_telling = 100
    # min_keywords = 100
    # min_light = 100
    # min_shadow = 100
    # min_q = 100
    # for card in data['cards']:
    #     try:
    #         a = json_Classification(card)
    #         min_fortune_telling = min(min_fortune_telling, len(a["fortune_telling"]))
    #         min_keywords = min(min_keywords, len(a["keywords"]))
    #         min_light = min(min_light, len(a["light"]))
    #         min_shadow = min(min_shadow, len(a["shadow"]))
    #         min_q = min(min_q, len(a["Questions to Ask"]))
    #     except:
    #         print(a)
    # print(min_fortune_telling, min_keywords, min_light, min_shadow, min_q)
    # df = read_csv()
    # for a, b, c, d in zip(df['Card 1'], df['Card 2'], df['Card 3'], df['Reading']):
    #     temp = tarot_prompt_generated(Card_1=a, Card_2=b, Card_3=c, Reading=d)
    #     print(f"\n\n")
    #     print(temp)
    train_data_generated(csv_file=os.path.join(ROOT, '塔羅', 'mbart_sc.csv'),
                         json_file=os.path.join(ROOT, '塔羅', 'Tarot Deck', 'GPT4o_sc.json'),
                         output_file=os.path.join(ROOT, 'output.csv'))

    get_data(file=os.path.join(ROOT, 'output.csv'))

    pass
