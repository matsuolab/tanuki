# tokenize

```
Tanuki_pretraining
    |
　~~~~~
    |-- tokenize
    |       |
    |       |-- README.md
    |       |
    |       |-- scripts : 必要なスクリプト
    |       |
    |       |-- tokenizer : 学習済みトークナイザーモデルの格納場所
    |       |
    |       |-- input_jsonl : トークナイズ対象の.jsonlの格納場所
    |       |
    |       |-- tokenized_data : トークナイズされたデータの保存場所
    |       |
    |       |-- logs : ログの保存場所
    |       |
    |       |-- tokenize_config.yaml : 設定.yaml
    |       |
    |       |-- tokenize_preprocess.py : 実体のスクリプト
    |       |
    |       |-- tokenize.sh : 実行シェルスクリプト
```

## 実行コマンド方法

tokenize.sh の7行目の
export PYTHONPATH=/home/ubuntu/Tanuki_pretraining/Megatron-LM:$PYTHONPATH
を自身の環境の Megatron-LM のパスに変更する

```bash
Tanuki_pretraining$ ./tokenize/tokenize.sh　　　##実行場所に注意
```

## tokenize_config.yaml について
```
#tokenize
input:                                  #トークナイズ対象の.jsonlを一つ上のディレクトリで指定
 - ./tokenize/input_jsonl/hanrei_1      　ディレクトリ構造を厳守
 - ./tokenize/input_jsonl/hanrei_2
 
#tokenize
input_tokenizer_file: ./tokenize/tokernizer/tokenizer_scale200.model    
output_prefix:  ./tokenize/tokenized_data/
seq_length: 2048                        #シーケンス数を指定
max_workers: 2                          #同時並列化する数を指定　CPU数に依存
```
