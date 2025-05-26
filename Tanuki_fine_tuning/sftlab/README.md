# sftlab
sftの実験コードおよびテンプレートを提供します。  
実験管理のため、一定のディレクトリ構造や命名規則に従って構成されています。

## Overview
### ディレクトリ構成
全体の構成は以下のようになっています。
```
sftlab
├── LICENSE
├── README.md
├── base_config   各実験で共通する設定を記載するconfigを格納する
├── experiments　　　　　　実験用コードを格納する
├── playground    experimentsに含めない実験コードを格納する(wandbのprojectは分離され、コードもgitignoreされるので自由に使って良い)
└── template　　　　　　　　　　　　実験用コードのテンプレートが格納されている
```
実験用コードは基本的に以下の構成に従って作ります。
```
experiments                  
└── your_project_name                         検証テーマごとに作成する(e.g. ハルシネーションの効果を検証する)
    └── your_exp_name                         学習コード(train.pyやrun.py)ごとに作成する
        ├── accelerate_config
        │   └── your_accelerate_config.yaml   accelerateを使用する場合の設定を記載したファイル
        ├── exp_config
        │   └── your_exp_config.yaml          実験設定(モデル、データ、学習パラメータなど)を記載したファイル
        ├── run.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　train.pyを実行するコード
        └── train.py                          sftを行う学習コード
```
projectとexpが1:n、expとrun.py/train.pyが1:1、expとexp_config.yaml/accelerate_config.yamlが1:nで紐づくようになっています。  
学習コードが異なるような実験はexpを分け、設定の違いはconfig.yamlで切り替えるようにしています。 

## Setup
condaで仮想環境を作成します。
```
# Python仮想環境を作成
conda create -n sft python=3.11 -y

# 作成したPython仮想環境を有効化
conda activate sft

# Python仮想環境を有効化した後は (python3コマンドだけでなく) pythonコマンドも使えることを確認
which python && echo "====" && python --version
```
リポジトリをgit cloneし、必要なライブラリをインストールします。
```
# リポジトリをgit clone
git clone https://github.com/team-hatakeyama-phase2/sftlab.git
cd sftlab

# 必要なライブラリのインストール
pip install -r requirements.in 

# この後のflash-attnでエラーになったので以下を実行
export LD_LIBRARY_PATH={path/to/your/miniconda3}/envs/sft/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

pip install flash-attn --no-build-isolation
pip install --upgrade accelerate
pip install datasets
```
base_configを設定します。ｗandbのentityやモデルの保存先など、全実験で共通して使われる設定になります。
```
# base_config_template.yamlをコピー
cp base_config/base_config_template.yaml base_config/base_config.yaml 

# base_config.yamlを以下のように編集
# 指定したディレクトリはあらかじめ作成しておくこと
output_dir: /path/to/your/output_dir (以下に実験のディレクトリが作成され、モデルが保存されます e.g. /storage5/personal/your name/)
hf_cache_dir: /path/to/your/hf_cache_dir (以下にhugginface datasetのcacheが保存されます e.g. /storage5/personal/your name/)
wandb:
  entity: your wandb entity name (e.g. weblab-geniac1)
```
必要に応じて実験コードを用意します。
```
# 該当するプロジェクトがない場合は、新しくプロジェクトディレクトリを作成する
mkdir experimets/your_project_name (zero埋め2桁数字_検証テーマがわかるプロジェクトのコード名にする予定 e.g. 01_hullucination)

# 新しく実験を始める場合や既存のrun.pyやtrain.pyと違うコードを使いたい場合は新しく実験ディレクトリを作成する
# template以下に実験ディレクトリのテンプレートが用意されているので、必要に応じてコピーして新しい実験ディレクトリとして使って良い
cp -r /path/to/target/template experimets/your_project_name/{exp_name} (特に制限はないが、コンフリクトが起きない名前が良い e.g. 作成者が識別できる名前_他の実験と区別できる名前)
```
モデルやデータ、パラメータを変えたいだけであれば、accelerate_configやexp_configに新しいconfigファイルを作成し、run.pyの実行時に指定すれば設定を切り替えられます。   
学習コードも変えたい場合は、新しい実験ディレクトリを作成した上でrun.pyやtrain.pyを編集します。(expとtrain.py/run.pyを1:1の関係に保つ)

## Running the experiment code
実験コードを実行します。
```
# $LD_LIBRARY_PATHの設定
export LD_LIBRARY_PATH={/path/to/your/miniconda3}/envs/sft/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# 実験ディレクトリに移動
cd  experimets/your_project_name/your_exp_name

# your_config.yamlの設定で実行
python run.py your_config.yaml

# マルチGPUを使用する場合
python run.py your_config.yaml --accelerate_config your_accelerate_config.yaml

# 実験管理用に追加でwandbのtagをつけたい場合
python run.py your_config.yaml --extra_tag tag1 tag2 tag3 (任意の個数設定可能)

# wandbにrunが増えすぎると見づらいため、一旦分離されたプロジェクト(sftlab-debug)に記録したい場合
python run.py your_config.yaml --debug
```

## Viewing results
学習されたモデルは`base_config.yaml`の`output_dir`で指定した場所にプロジェクト名に応じたディレクトリが作成され、保存されます。  
```
# 保存先のディレクトリ
{/path/to/output_dir/in/base_config.yaml}/sftlab-{parent_dir_of_your_project}/{your_project_name}/{your_exp_name}-{your_exp_config}

# acceraletareを使用した場合
{/path/to/output_dir/in/base_config.yaml}/sftlab-{parent_dir_of_your_project}/{your_project_name}/{your_exp_name}-{your_exp_config}-{your_acceralete_config}

# run.py実行時に--debugを指定した場合
{/path/to/output_dir/in/base_config.yaml}/sftlab-debug/{your_project_name}/{your_exp_name}-{your_exp_config}
```

lossなどの情報は`base_config.yaml`の`wandb:entity`で指定したwandbに記録されます。  

project
```
# --debugをつけてrun.pyを実行した場合
sftlab-debug

# playground以下のprojectでrun.pyを実行した場合
sftlab-playground

# experiments以下のprojectでrun.pyを実行した場合
sftlab-experiments-your_project_name
```

run name
```
# accelerateを使用しない場合
your_exp_name-your_exp_config

#　accelerateを使用した場合
your_exp_name-your_exp_config-your_accelerate_config
```

その他
- exp_nameがgroupとして登録されるようになっています
- 実行に使用したコード(run.py, train.py, your_exp_config.yaml, 使用していればyour_accelerate_config.yaml)がartifactのscriptsに保存されます
- run.py実行時に--extra_tagでtagを付与することができます、必要に応じて実験管理にお使いください


