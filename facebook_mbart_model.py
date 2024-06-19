from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from functools import lru_cache
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
import json
import sys
import os
import numpy as np

# => "The Secretary-General of the United Nations says there is no military solution in Syria."
# Arabic(ar_AR), Czech(cs_CZ), German(de_DE), English(en_XX), Spanish(es_XX), Estonian(et_EE), Finnish(fi_FI), French(
#     fr_XX), Gujarati(gu_IN), Hindi(hi_IN), Italian(it_IT), Japanese(ja_XX), Kazakh(kk_KZ), Korean(ko_KR), Lithuanian(
#     lt_LT), Latvian(lv_LV), Burmese(my_MM), Nepali(ne_NP), Dutch(nl_XX), Romanian(ro_RO), Russian(ru_RU), Sinhala(
#     si_LK), Turkish(tr_TR), Vietnamese(vi_VN), Chinese(zh_CN), Afrikaans(af_ZA), Azerbaijani(az_AZ), Bengali(
#     bn_IN), Persian(fa_IR), Hebrew(he_IL), Croatian(hr_HR), Indonesian(id_ID), Georgian(ka_GE), Khmer(
#     km_KH), Macedonian(mk_MK), Malayalam(ml_IN), Mongolian(mn_MN), Marathi(mr_IN), Polish(pl_PL), Pashto(
#     ps_AF), Portuguese(pt_XX), Swedish(sv_SE), Swahili(sw_KE), Tamil(ta_IN), Telugu(te_IN), Thai(th_TH), Tagalog(
#     tl_XX), Ukrainian(uk_UA), Urdu(ur_PK), Xhosa(xh_ZA), Galician(gl_ES), Slovene(sl_SI)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if sys.platform == 'darwin':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 10
max_length = 150

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device,
                                                                                                     non_blocking=True)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", force_download=True)
tokenizer.src_lang = "en_XX"


def module_setting(data):
    encoded_sent = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=data,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,  # 添加截断参数
        return_tensors='pt'  # 返回 PyTorch 张量
    )

    # Convert lists to tensors

    input_ids = encoded_sent['input_ids']
    attention_masks = encoded_sent['attention_mask']

    return TensorDataset(input_ids, attention_masks)


# def module_setting(data):
#     encoded_sent = tokenizer(data,
#                              truncation=True,
#                              max_length=max_length,
#                              padding='max_length',
#                              return_tensors="pt")
#     input_ids = encoded_sent['input_ids']
#     attention_mask = encoded_sent['attention_mask']
#     return TensorDataset(input_ids, attention_mask)

# 德語(de_DE) 法語(fr_XX) 荷蘭語(nl_XX) 瑞典語(sv_SE) 日文(ja_XX)
# German(de_DE)、(fr_XX)、(nl_XX)、(sv_SE)
def model_eval(data):
    output = []

    dataset = module_setting(data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model.eval()

    for step, batch in enumerate(tqdm(dataloader)):
        input_ids, attention_mask = tuple(t.to(device, non_blocking=True) for t in batch)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_length,
                                 forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])

        # 解碼並打印每個輸出
        for line in out:
            # summary = tokenizer.decode(line, skip_special_tokens=True)
            output.append(line)

    for i, line in enumerate(output):
        summary = tokenizer.decode(line, skip_special_tokens=True)
        output[i] = summary

    return output


if __name__ == '__main__':
    pass
