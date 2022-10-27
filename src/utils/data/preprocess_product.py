import gc
import os
import time
from tqdm import tqdm
import dgl
import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer

import utils.function as uf
from utils.function.dgl_utils import sample_nodes
from utils.settings import *


def _load_ogb_products(d, labels):
    from ogb.utils.url import download_url, extract_zip
    # Get Raw text path

    import gdown
    opath = os.path.join(d.raw_text_url, d.data_root)
    print(os.path.join(opath, "Amazon-3M.raw"))
    output = os.path.join(opath, 'Amazon-3M.raw.zip')
    if not os.path.exists(os.path.join(opath, "Amazon-3M.raw.zip")):
        url = d.raw_text_url
        gdown.download(url=url, output=output, quiet=False, fuzzy=False)
    if not os.path.exists(os.path.join(opath, "Amazon-3M.raw")):
        extract_zip(output, opath)
    raw_text_path = os.path.join(opath, "Amazon-3M.raw")

    def read_mappings(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2productcategory.csv.gz"
        products_asin_path_csv = f"{data_root}/mapping/nodeidx2asin.csv.gz"  #
        products_ids = pd.read_csv(products_asin_path_csv)
        categories = pd.read_csv(category_path_csv)
        # categories.columns = ["ID", "category"]  # 指定ID 和 category列写进去
        return categories, products_ids  # 返回类别和商品ID

    def get_mapping_product(labels, meta_data: pd.DataFrame, products_ids: pd.DataFrame, categories):
        # ! Read mappings for OGBN-products
        products_ids.columns = ["ID", "asin"]
        categories.columns = ["label_idx", "category"]  # 指定ID 和 category列写进去
        meta_data.columns = ['asin', 'title', 'content']
        products_ids["label_idx"] = labels
        data = pd.merge(products_ids, meta_data, how="left", on="asin")  # ID ASIN TITLE
        data = pd.merge(data, categories, how="left", on="label_idx")  # 改写是为了拼接到一起
        # ID ASIN LABEL_IDX TITLE CATEGORY
        return data

    def read_product_json(raw_text_path):
        import json
        import gzip
        if not os.path.exists(os.path.join(raw_text_path, "trn.json")):
            trn_json = os.path.join(raw_text_path, "trn.json.gz")
            trn_json = gzip.GzipFile(trn_json)
            open(os.path.join(raw_text_path, "trn.json"), "wb+").write(trn_json.read())
            os.unlink(os.path.join(raw_text_path, "trn.json.gz"))
            tst_json = os.path.join(raw_text_path, "tst.json.gz")
            tst_json = gzip.GzipFile(tst_json)
            open(os.path.join(raw_text_path, "tst.json"), "wb+").write(tst_json.read())
            os.unlink(os.path.join(raw_text_path, "tst.json.gz"))
            os.unlink(os.path.join(raw_text_path, "Yf.txt"))  # New

        i = 1
        for root, dirs, files in os.walk(os.path.join(raw_text_path, '')):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r', encoding='utf_8_sig') as file_in:
                    title = []
                    for line in file_in.readlines():
                        dic = json.loads(line)

                        dic['title'] = dic['title'].strip("\"\n")
                        title.append(dic)
                    name_attribute = ["uid", "title", "content"]
                    writercsv = pd.DataFrame(columns=name_attribute, data=title)
                    writercsv.to_csv(os.path.join(raw_text_path, f'product' + str(i) + '.csv'), index=False,
                                     encoding='utf_8_sig')  # index=False不输出索引值
                    i = i + 1
        return

    def read_meta_product(raw_text_path):
        # 针对read_meta_data
        if not os.path.exists(os.path.join(raw_text_path, f'product3.csv')):
            read_product_json(raw_text_path)  # 弄出json文件
            path_product1 = os.path.join(raw_text_path, f'product1.csv')
            path_product2 = os.path.join(raw_text_path, f'product2.csv')
            pro1 = pd.read_csv(path_product1)
            pro2 = pd.read_csv(path_product2)
            file = pd.concat([pro1, pro2])
            file.drop_duplicates()
            file.to_csv(os.path.join(raw_text_path, f'product3.csv'), index=False, sep=" ")
        else:
            file = pd.read_csv(os.path.join(raw_text_path, 'product3.csv'), sep=" ")

        return file

    print('Loading raw text')
    meta_data = read_meta_product(raw_text_path)  # 二维表
    categories, products_ids = read_mappings(d.data_root)
    node_data = get_mapping_product(labels, meta_data, products_ids, categories)  # 返回拼接后的数据
    import gc
    del meta_data, categories, products_ids
    text_func = {
        'T': lambda x: x['title'],
        'TC': lambda x: f"Title: {x['title']}. Content: {x['content']}",
    }
    node_data['text'] = node_data.apply(text_func[d.process_mode], axis=1)
    node_data['text'] = node_data.apply(lambda x: ' '.join(str(x['text']).split(' ')[:d.cut_off]), axis=1)
    node_data = node_data[['ID', 'text']]  # 最后节点只剩ID和处理后的text
    return node_data


def _tokenize_ogb_product(d, labels):
    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
    node_text = _load_ogb_products(d, labels)
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    # ! Determine the least memory cost type
    node_chunk_size = 1000000
    max_length = 512
    shape = (d.md['n_nodes'], max_length)
    # tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
    token_keys = tokenizer(get_text([0]), padding='max_length', truncation=True, max_length=512).data.keys()
    x = {k: np.memmap(d.info[k].path, dtype=d.info[k].type, mode='w+', shape=d.info[k].shape)
         for k in token_keys}

    for i in tqdm(range(0, shape[0], node_chunk_size)):
        j = min(i + node_chunk_size, shape[0])
        tokenized = tokenizer(get_text(range(i, j)), padding='max_length', truncation=True, max_length=512).data
        for k in token_keys:
            x[k][i:j] = np.array(tokenized[k], dtype=d.info[k].type)
    uf.pickle_save('processed', d._processed_flag['token'])
