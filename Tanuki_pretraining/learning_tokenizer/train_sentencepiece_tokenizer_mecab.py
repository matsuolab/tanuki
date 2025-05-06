# ## mecabで分かち書きしながら、tokenizerを作成する


BASE_PATH = "/home/ubuntu/Tanuki_pretraining/learning_tokenizer" #自身の作業用フォルダーを指定

# %%
!sudo apt install -y mecab libmecab-dev mecab-ipadic-utf8

# %% [markdown]
# jsonlファイルをtxtファイルに変換

# %%
import json
target_path="./learning_corpus/corpus_scale_200.jsonl" #自身で用意したコーパス
num = 0
max_len = 0
with open("./preprocess_data/corpus_text.txt","w") as o:
    o.write("")
with open("./preprocess_data/corpus_text.txt","a") as o:
    with open(target_path, "r") as f:
        for line in f:
            num += 1
            jsonl_line = (json.loads(line)["text"])
            o.write(jsonl_line + "\n")
            if len(jsonl_line) > max_len:
                max_len = len(jsonl_line)
                max_line = jsonl_line

print(f'num: {num}, max_len: {max_len}' )

# %%
print (max_line)

# %%
#わかちがき
!mecab -F"%M||||" -E"\n" -b 400000 < ./preprocess_data/corpus_text.txt  > ./preprocess_data/corpus_text.tok
#!mecab -F"%M||||" -E"\n" -b 100000 < ./preprocess_data/corpus_text.txt  > ./preprocess_data/corpus_text_100K.tok

# %%
!head -n100 ./preprocess_data/corpus_text.tok

# %%
from types import SimpleNamespace
import sentencepiece as spm
import sys
import yaml
import os
sys.path.append(BASE_PATH)
from config.special_token_list import *

def yaml_to_namespace(yaml_path):
    with open(yaml_path, 'r') as file:
        # YAMLファイルを辞書として読み込む
        data = yaml.safe_load(file)
        # 辞書をSimpleNamespaceに変換
        return recursive_namespace(data)

def recursive_namespace(data):
    if isinstance(data, dict):
        # 再帰的に辞書の各要素をSimpleNamespaceに変換
        return SimpleNamespace(**{k: recursive_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        # リストの要素も変換
        return [recursive_namespace(v) for v in data]
    else:
        # その他のデータ型はそのまま返す
        return data

args = yaml_to_namespace('./config/config.yaml')

#args.input="./preprocess_data/corpus_text.tok"
#args.vocab_size=3000

# %%
spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        num_threads=args.num_threads,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
        user_defined_symbols=[
            BOS_TOKEN,
            EOS_TOKEN,
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            EOD_TOKEN,
            MASK_TOKEN,
            NEWLINE_TOKEN,
            EXTRA_TOKEN1,
            EXTRA_TOKEN2,
            EXTRA_TOKEN3,
            EXTRA_TOKEN4,
        ],  # Note: `NEWLINE_TOKEN` is needed in `user_defined_symbols`.
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        pretokenization_delimiter="||||",
        # google_colab用
        input_sentence_size=1000000,
    )

# %%
import sentencepiece as spm
#test
model_path="./tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=model_path)

# %%

text=" 価格詳細, 日本の正式代理店で購入する必要がある"
text="he is a good man"
text="import pandas as pd  from tqdm import tqdm"
text="""
def yaml_to_namespace(yaml_path):
    with open(yaml_path, 'r') as file:
        # YAMLファイルを辞書として読み込む
        data = yaml.safe_load(file)
        # 辞書をSimpleNamespaceに変換
        return recursive_namespace(data)
"""

text="これはテストです。"
text="吾輩は猫である｡名前はまだない"
text="吾輩は猫である｡\n名前はまだない"
(sp.encode(text, out_type=str)),(sp.encode(text))

# %%
t="ああ\nああ"
tokens=sp.encode(t)
t,tokens,sp.decode(tokens)


