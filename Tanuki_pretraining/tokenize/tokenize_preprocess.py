# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
from datetime import datetime
import pytz
import pandas as pd

# 日本のタイムゾーンを設定
japan_timezone = pytz.timezone('Asia/Tokyo')

# 現在の日本時間を取得
start_japan = datetime.now(japan_timezone)    
# 日時を秒単位までのフォーマットで表示
formatted_time = start_japan.strftime("%Y-%m-%d %H:%M:%S")
print("Start time: ", formatted_time)

import gzip
import glob
# import torch
import numpy as np
from tqdm import tqdm

import multiprocessing
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

import concurrent.futures

from megatron.training.tokenizer import build_tokenizer
from scripts import indexed_dataset


import logging
from logging.handlers import TimedRotatingFileHandler

# ロガーの設定関数
def setup_logger(name, log_file, level=logging.INFO):
    # 日本時間のタイムゾーンを設定
    japan_timezone = pytz.timezone('Asia/Tokyo')
    start_japan = datetime.now(japan_timezone)
    formatted_time = start_japan.strftime("%Y-%m-%d_%H-%M-%S")
    
    # ログファイルの設定
    log_filename = f"./tokenize/logs/Tokenizing_{log_file}log_{formatted_time}.log"

    # ロガーを設定
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ローテーティングハンドラを設定
    handler = TimedRotatingFileHandler(log_filename, when="midnight", interval=1)
    handler.suffix = "%Y-%m-%d"
    
    # フォーマッタを設定
    class JSTFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, japan_timezone)
            if datefmt:
                return dt.strftime(datefmt)
            else:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    formatter = JSTFormatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    
    # ハンドラをロガーに追加
    logger.addHandler(handler)
    
    return logger

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    tokenizer = None 
    def __init__(self, args):
        self.args = args
        self.seq_length = args.seq_length 

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        ids = {}
        lens = {}
        try:
            data = json.loads(json_line)
        except json.decoder.JSONDecodeError as e:
            print(f"デコードエラー: {e} ") 
            json_line = ""
            text = ""
            return ids, lens, len(json_line), text
        for key in self.args.json_keys:
            if key != "text" and key not in data:
                text = ""
                continue  # キーがない場合、このイテレーションをスキップ
            try:
                text = data[key]
            except:
                text = ""
                continue  # キーがない場合、このイテレーションをスキップ
            # print( "text" , text[0],":" , text[1],":" , text[-2],":" , text[-1] )
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line), text

class Process(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def count_lines(self, file_path):
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs='+', required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--seq-length', type=int, default=2048,
                       help='Setting seq-length')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--max_workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (max_workers * partitions) = available CPU cores.'))

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names

def check_files_exist(target_files_list, key, num_partitions):
    print("L237 num_partitions" , num_partitions)
    for i in range(num_partitions):
        print( target_files_list[i][key]) 
        if not os.path.exists(target_files_list[i][key]):
            return False
    return True

def count_lines(file_path):
    try :
        with open(file_path, 'r') as file:
            return sum(1 for _ in file)
    except:
        print( f"L221 Warning!  {file_path} is not utf-8. count_lines is dummy."  )
        return 10^6

def delete_cache(directory):
    # ディレクトリを下降順で走査
    for root, dirs, files in os.walk(directory, topdown=False):
        # ファイルの削除
        for name in files:
            file_path = os.path.join(root, name)
            if not (name.endswith('.bin') or name.endswith('.idx')):
                os.remove(file_path)
                # print(f"Deleted file: {file_path}")

        # 空のディレクトリの削除
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
                # print(f"Deleted directory: {dir_path}")
            except OSError as e:
                print(f"Directory not empty: {dir_path}")

def index_to_alphabet(index):
    alphabet = ""
    while index >= 0:
        alphabet = chr(97 + index % 26) + alphabet
        index = index // 26 - 1
    return alphabet

# 並列を行う関数
def process_item(target_files_list):
    args = target_files_list['args']
    seq_length = args.seq_length 
    process_idx_alphabet = index_to_alphabet( target_files_list["process_idx"])
    logger = target_files_list['main_logger']
    input_file_name = target_files_list['partition']
    output_prefix = target_files_list['output_prefix']
    logger.info(f"Process start: {process_idx_alphabet}-{input_file_name}-{seq_length} ")

    level = "document"
    if target_files_list['sentence_split']:
        level = "sentence"

    encoder = Encoder(args)
    encoder.initializer()

    tokenizer = build_tokenizer(args)
    ids = {}

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    token_total_num = 0
    char_total_num = 0

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                        key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                        key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                impl=args.dataset_impl,
                                                vocab_size=tokenizer.vocab_size)

    total = target_files_list["jsonl_file_size_gb"] * 1024**3
    index_alphabet = index_to_alphabet( target_files_list["process_idx"])
    thresholds = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.0, 2.5]  # 進捗を表示する閾値
    next_threshold = 0

    line_count = 0
    error_count = 0
    total_bytes = 0
    with open(input_file_name, 'r', encoding='utf-8', errors='ignore') as file:
        for i, line in enumerate(file):
            try:
                # 行をUTF-8でエンコードしてバイト数を計算
                line_bytes = len(line.encode('utf-8'))
            except UnicodeEncodeError as e:
                print(f"Encoding error at line {i+1}: {e}. This line will be skipped.　In {file}")
                continue  # 次の行へスキップ
            total_bytes += line_bytes
            progress = total_bytes/ total
            if progress >= thresholds[next_threshold]:
                print(f"Progress [P-{index_alphabet}] : {int(progress * 100)}% complete.")
                logger.info(f"Progress [P-{index_alphabet}] : {int(progress * 100)}% complete.")
                if next_threshold < len(thresholds) - 1:
                    next_threshold += 1

            ids, sentence_lens, len_json_line, text =  encoder.encode(line)
            if text == "":
                print(f"Warning: text is None. '{input_file_name}' is not good. Line {line_count} is skipped.")
                # error_count += 1
                if error_count > 100:
                    print(f"Error: '{input_file_name}' is skipped.")
                    break
            for key in args.json_keys:
                if key not in ids:
                    print(f"Warning: Key '{key}' not found in {input_file_name}.")
                    continue  # キーがない場合、このイテレーションをスキップ
                token = ids['text'] #31番始まり、7番終わりのトークンのdict
                # encoded_docs.extend(token)
                token_num = len(token) #トークン数
                char_num = len(text) #文字数
                token_total_num += token_num
                char_total_num += char_num
            for key in ids.keys():
                builders[key].add_doc(ids[key], sentence_lens[key])
            line_count += 1

    time.sleep(20)
    builders[key].finalize(output_idx_files[key])
    time.sleep(20)
    delete_cache( output_prefix )
    bin_file_size_b = os.path.getsize(output_bin_files[key]) 
    idx_file_size_b = os.path.getsize(output_idx_files[key]) 
    idx_file_size_mb = 0 if idx_file_size_b == 0 else idx_file_size_b / (1024 ** 2)
    bin_file_size_gb = 0 if bin_file_size_b == 0 else bin_file_size_b / (1024 ** 3)
    file_size_ratio = 0 if bin_file_size_b == 0 else bin_file_size_b / token_total_num 

    if file_size_ratio == 2:
        file_size_ratio_OK = True
    else:
        file_size_ratio_OK = False

    token_total_num = 0 if token_total_num == 0 else token_total_num /10**9
    char_total_num = 0 if char_total_num == 0 else char_total_num /10**9
    result_dict = {
        'input_file_name': input_file_name, 
        'output_bin_files': output_bin_files[key] , 
        'token_total_[BT]': token_total_num,
        'char_total_[BW]': char_total_num,
        'jsonl_file_size_[GB]': target_files_list['jsonl_file_size_gb'] ,
        'bin_file_size_[GB]': bin_file_size_gb ,
        'bin_file_size_ratio': file_size_ratio ,
        'file_size_ratio_OK': file_size_ratio_OK ,
        'idx_file_size[MB]': idx_file_size_mb ,
        'tokenizer_model':args.tokenizer_model }

    df = pd.DataFrame( [result_dict] )
    # CSVに出力
    csv_path = "./{}_result.csv".format(output_prefix)
    df.to_csv(csv_path, index=False)
    logger.info(f"Process Finish: {process_idx_alphabet}-{result_dict} ")

    return result_dict

def main():
    args = get_args()

    # 現在の日本時間を取得
    start_japan = datetime.now(japan_timezone)    
    # 日時を秒単位までのフォーマットで表示
    formatted_time = start_japan.strftime("%Y-%m-%d %H:%M:%S")
    print("Start time: ", formatted_time)
    print("args.seq_length: ", args.seq_length)

    logger = setup_logger('main_logger', '', level=logging.INFO)
    logger.info(f"Start time: {formatted_time}" )
    logger.info(f"Set sequence length is : {args.seq_length}")
        
    extension = ".jsonl"

    #処理するファイルリスト
    target_dir_list = (args.input )
    # JSONライブラリを使用して文字列をリストに変換
    target_dir_list = target_dir_list[1:-1]

    # リスト内の各ディレクトリに対して操作を行う
    list_dir_path = []
    for target_dir in target_dir_list:
        # 余分なクォートとカンマを削除
        target_dir = (target_dir.strip('",'))
        if not os.path.isdir(target_dir):
            print(f"Directory {target_dir} does not exist. Skipping...")
            continue  # ディレクトリが存在しない場合はスキップ
        list_dir_path.append(target_dir)
    
    print("L356 list_dir_path" , list_dir_path)
    #処理するファイルリスト
    list_files_in_dir = []
    target_files_list = []
    list_jsonl_files_path_in_dir = []
    list_jsonl_files_name_in_dir = []
    list_dir_name = []
    for dir_path in list_dir_path:
        print("dir_path ", dir_path)
        # ディレクトリが存在するか確認

        # .jsonlファイルをリスト化
        jsonl_files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl')]
        list_jsonl_files_name_in_dir.append( jsonl_files )
        # フルパスでリスト化する場合
        jsonl_files_full_path = [os.path.join(dir_path, f) for f in jsonl_files]
        list_jsonl_files_path_in_dir.append(jsonl_files_full_path)
        # 結果を出力
        print(jsonl_files_full_path)
        list_dir_name.append( os.path.basename(dir_path)  )
        # 現在のディレクトリ内のファイルをリストアップ
        # list_files_in_dir = os.listdir(dir)
        # for file_in_dir in list_files_in_dir:
        #     list_jsonl_files_path_in_dir.append( os.path.join(dir , file_in_dir)  )
        # print("list_jsonl_files_path_in_dir" , list_jsonl_files_path_in_dir)
        # list_jsonl_files_name_in_dir.append( [file for file in list_jsonl_files_path_in_dir if file.endswith(extension)] )
        # list_dir_name.append( os.path.basename(dir) )

    # ファイルリストを表示
    print("list_files_in_dir" , list_files_in_dir)
    print("list_jsonl_files_path_in_dir" , list_jsonl_files_path_in_dir)
    print("list_jsonl_files_name_in_dir" , list_jsonl_files_name_in_dir)
    print("list_dir_name" , list_dir_name)

    for index, dir_path in enumerate(list_dir_path):
        print("index" , index )
        dir_name = list_dir_name[index]
        for p_idx , file_name in enumerate(list_jsonl_files_name_in_dir[index]):
            file_path = os.path.join(dir_path , file_name)
            print("file_path", file_path)

            # ファイルサイズを取得し、ギガバイトに変換
            file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
            file_name_only, extension = os.path.splitext(file_name)
            sentence_split_file = file_name + "_ss" + extension
            output_dir = os.path.join(args.output_prefix, dir_name) 
            output_prefix = os.path.join(output_dir, file_name_only)

            # フォルダが存在しないことを確認し、存在する場合はエラーを発生
            assert not os.path.exists(output_prefix), f"Please delete the folder at {output_prefix} and rerun the program."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            logger.info(f"Process_idx: {p_idx}, file_path: {file_path} ")
            file_dict = {
                'process_idx': p_idx,
                'partition': file_path, 
                'sentence_split': sentence_split_file,
                'output_prefix': output_prefix,
                'jsonl_file_size_gb': file_size_gb, 
                'main_logger':logger,
                'args':args }
            target_files_list.append(file_dict)

    list_results =[]
    #デバッグ用に並列か化処理をしない
    # for  target_files in target_files_list:
    #     list_results.append(process_item(target_files))
    # ProcessPoolExecutor を使用し、最大ワーカー数をmax_workersに設定
    # map を使用してリスト内の各ファイルに対して process_item 関数を並行実行
    with concurrent.futures.ProcessPoolExecutor(max_workers= args.max_workers) as executor:
        list_results = list(executor.map(process_item, target_files_list))

    print( "ProcessPoolExecutor Done. Please wait ...")
    logger.info(f"ProcessPoolExecutor Done. Please wait ...")

    time.sleep(10)
    print( "Please wait ...")
    time.sleep(10)
    print( "Make result ...")

    list_data_path = []
    total_BT = 0
    total_size_GB = 0
    for result in list_results:
        output_bin_path = result["output_bin_files"]
        data_path, extension = os.path.splitext(output_bin_path)
        list_data_path.append( str( '"' + data_path + '"') )
        total_BT += result["token_total_[BT]"]
        total_size_GB += result["jsonl_file_size_[GB]"]

    # 現在の日時を取得し、ファイル名に使用する形式にフォーマットする
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ファイル名に現在の日時を追加
    filename = f"./tokenize/tokenized_data/DATA_PATH_LIST_{current_time}.txt"
    # ファイルに書き出す
    with open( filename, 'w') as file:
        for path in list_data_path:
            file.write(path + '\n')  # 各パスを改行文字とともに書き出す                      

    df = pd.DataFrame(list_results)
    # CSVに出力
    csv_path =  f"{args.output_prefix}_{current_time}_{args.seq_length}_result.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"list_results: {list_results}")

    # 現在の日本時間を取得
    finish_japan = datetime.now(japan_timezone)    
    # 日時を秒単位までのフォーマットで表示
    formatted_time = finish_japan.strftime("%Y-%m-%d %H:%M:%S")
    print("Finish time: ", formatted_time)
    process_time = finish_japan - start_japan
    print( " ---- Finish   process_time : " , process_time)

    logger.info(f"Total Token [BT]: {total_BT}" )
    logger.info(f"Total jsonl size [GB]: {total_size_GB}" )
    logger.info(f"Finish time: {formatted_time}" )
    logger.info(f"Process_time: {process_time}" )

if __name__ == '__main__':
    main()

