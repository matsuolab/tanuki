import glob
import os
import random
import argparse
import time

import yaml


# argparseのパーサーを作成
parser = argparse.ArgumentParser(description='引数を取るサンプルスクリプト')
parser.add_argument('exp_config', type=str, help='実験に用いるモデル、データ、パラメータの設定ファイル')
parser.add_argument('--accelerate_config', type=str, default=None, help='deepspeedの設定ファイル')
parser.add_argument('--master_port', type=int, default=None, help='マスターのポート番号')
parser.add_argument('--hostfile', type=str, default=None, help='DeepSpeedのホストファイル')
parser.add_argument('--extra_tag', type=str, nargs='+', default=None, help='wandbの実験管理用に追加で付けるtag')
parser.add_argument('--debug', action='store_true', help='デバッグ用にwandbの記録を全てsftlab-debugに送る')

args = parser.parse_args()

# 置かれているディレクトリから実験名などを抜き出す
run_file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(run_file_path)
exp_name = os.path.basename(exp_dir)
exp_project_dir = os.path.dirname(exp_dir)
exp_project_name = os.path.basename(exp_project_dir)
sftlab_dir = os.path.dirname(exp_project_dir)
sftlab_dir_name = os.path.basename(sftlab_dir)

# 実験共通となる設定のyamlファイルを読み取る
base_config_path = os.path.dirname(sftlab_dir) + '/base_config/base_config.yaml'
with open(base_config_path) as file:
    base_config = yaml.safe_load(file)
print(base_config)

# 実験設定のyamlファイルを読み取る
exp_config_path = exp_dir + '/exp_config/' + args.exp_config
with open(exp_config_path) as file:
    exp_config = yaml.safe_load(file)
print(exp_config)


# wandbの設定
wandb_entity = base_config['wandb']['entity']
if args.debug:
    wandb_project = 'sftlab-debug'
else:
    wandb_project = f'sftlab-{sftlab_dir_name}'
    if sftlab_dir_name == 'experiments':
        wandb_project = wandb_project + f'-{exp_project_name}'
wandb_group = exp_name
wandb_name = f'{exp_name}-{args.exp_config.replace(".yaml", "")}'
if args.accelerate_config is not None:
    # wandb_name = wandb_name + '-' + args.accelerate_config.replace(".yaml", "")
    wandb_name = wandb_name + '-' + os.path.splitext(args.accelerate_config)[0]
wandb_tags = [args.exp_config, args.accelerate_config]
if args.extra_tag is not None:
    wandb_tags = wandb_tags + args.extra_tag
wandb_tags = ','.join([tag for tag in wandb_tags if tag is not None])

wandb_options=f" \
--wandb_entity {wandb_entity} \
--wandb_project {wandb_project} \
--wandb_group {wandb_group} \
--wandb_name {wandb_name} \
--wandb_tags {wandb_tags} \
"

# wandbのartifaceに保存する実験に用いたファイルパスを集める
save_scripts = [run_file_path, exp_config_path]

if args.accelerate_config is not None:
    # マルチgpu
    accelerate_config_path = exp_dir + '/accelerate_config/' + args.accelerate_config
    save_scripts.append(accelerate_config_path)
    if args.master_port is not None and args.hostfile is not None:
        # マルチノード
        pre_cmd = f"deepspeed --master_port {args.master_port} --hostfile {args.hostfile} {exp_dir}/train.py --deepspeed {accelerate_config_path}"    
    else:
        # シングルノード・マルチgpu
        pre_cmd=f"accelerate launch --config_file {accelerate_config_path} {exp_dir}/train.py"
else:
    #シングルgpu
    pre_cmd=f"python {exp_dir}/train.py"

save_scripts = ','.join(save_scripts)

# モデルの保存先
if args.debug:
    output_dir = 'sftlab-debug'
else:
    output_dir = f'sftlab-{sftlab_dir_name}'
output_dir = output_dir + f'/{exp_project_name}'
output_dir = base_config['output_dir'] + '/' + output_dir + '/' + wandb_name


exp_params = exp_config['exp_params']

# TODO: instruction_templateの改行がyamlから意図した形で渡せないのを修正する
cmd = f"""{pre_cmd}  \
    --num_train_epochs {exp_params['num_train_epochs']} \
    --per_device_train_batch_size {exp_params['per_device_train_batch_size']} \
    --per_device_eval_batch_size {exp_params['per_device_eval_batch_size']} \
    --gradient_accumulation_steps {exp_params['gradient_accumulation_steps']} \
    --save_strategy "{exp_params['save_strategy']}" \
    --save_steps {exp_params['save_steps']} \
    --logging_steps {exp_params['logging_steps']} \
    --learning_rate {exp_params['learning_rate']} \
    --warmup_ratio {exp_params['warmup_ratio']} \
    --lr_scheduler_type {exp_params['lr_scheduler_type']} \
    --{exp_params['dtype']} \
    --model_name_or_path {exp_config['model']['name']} \
    --use_fast {exp_params['use_fast']} \
    --output_dir {output_dir} \
    --gradient_checkpointing {exp_params['gradient_checkpointing']} \
    --max_seq_length {exp_params['max_seq_length']} \
    --use_peft {exp_params['use_peft']} \
    --peft_target_model "{exp_params['peft_target_model']}" \
    --use_flash_attention_2 {exp_params['use_flash_attention_2']} \
    --peft_lora_r {exp_params['peft_lora_r']} \
    --peft_lora_alpha {exp_params['peft_lora_alpha']} \
    --peft_lora_dropout {exp_params['peft_lora_dropout']} \
    --eval_strategy  {exp_params['eval_strategy']} \
    --eval_steps {exp_params['eval_steps']} \
    --hf_cache_dir {base_config['hf_cache_dir']} \
    --save_scripts {save_scripts} \
    --exp_config_path {exp_config_path} \
    --do_eval {exp_params['do_eval']} \
    """

if exp_params['neftune_noise_alpha'] is not None and exp_params['neftune_noise_alpha'] != 'None':
    cmd += f"--param_neftune_noise_alpha {exp_params['neftune_noise_alpha']} \
    "

if exp_config['tokenizer']['name'] is not None:
    cmd += f"--tokenizer_name_or_path {exp_config['tokenizer']['name']} \
    "

cmd += wandb_options

os.system(cmd)