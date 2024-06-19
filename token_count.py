import tiktoken
import os

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"


def load_txt(test_name):
    prompt = []
    label = []
    with open(os.path.join('test', test_name), 'r', encoding='utf-8') as f:
        text = f.readlines()
        for line in text:
            line = line.split('^')
            prompt.append(line[0].strip())
            label.append(line[1].strip())

    return [prompt, label]


input_token_not_text = 310
offset = 4  # gpt-4o固定+4
# To get the tokeniser corresponding to a specific model in the OpenAI API:

cged_14_15_20_pku = load_txt('cged_14_15_20_pku.txt')
answer_sample = load_txt('answer_sample.txt')

# enc = tiktoken.encoding_for_model("gpt-4")
enc = tiktoken.encoding_for_model("gpt-4o")

'''
input_token_all = sum(input_token_not_text + len(enc.encode(i)) for i in cged_14_15_20_pku[0])
output_token_all = sum(output_token for i in cged_14_15_20_pku[0])

print(input_token_all)
print(output_token_all)
'''

input_token_all = sum(input_token_not_text + len(enc.encode(i)) for i in answer_sample[0])
output_token_all = sum(len(enc.encode(i)) + offset for i in answer_sample[0])

# print(len(enc.encode('生活的活力也不比以前。')))  # 9
# print(len(enc.encode('精品商店是上海最繁荣最流行的服装店之一，而且它们的进口服装也很丰富。')))  # 27
# print(len(enc.encode('当时我在管理花园。')))  # 8
print(input_token_all)
print(output_token_all)

# completion_tokens 模型生成了 x 個 token 13 31 12
# prompt_tokens 輸入了 x 個 token 319 339 318

# a = '''生活的活力也不比以前。'''  # 11
# b = '''生活的活力也不比以前少。'''  # 12
# c = '''精品商店是上海的最繁荣最留行的服装店之一，而且它们的近口服装也风丰。'''  # 43
# d = '''精品商店是上海最繁榮、最有名氣的服裝店之一，而且它們的進口服裝也非常豐富。'''  # 51
# e = '''精品商店是上海最繁荣最流行的服装店之一，而且它们的进口服装也很丰富。''' 43
#
# print(len(enc.encode(a)))
# print(len(enc.encode(b)))
# print(len(enc.encode(c)))
# print(len(enc.encode(d)))
