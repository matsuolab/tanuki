# training

```
Tanuki_pretraining
    |
　~~~~~
    |-- training
    |       |
    |       |-- README.md
    |       |
    |       |-- tokenizer : トークナイザーの保存場所
    |       |
    |       |-- training_corpus : 学習コーパスの格納場所
    |       |
    |       |-- checkpoints : チェックポイントの保存場所 
    |       |
    |       |-- tools : TP、PP変換に使うスクリプト
    |       |
    |       |-- one_RTX3090_test.sh    
    |       |
    |       |-- 8B-A100_4_pretraining_initial_v2.sh
    |       |
    |       |-- 8B-A100_4_pretraining_continue_v2.sh
```

## デバッグ用（超軽量モデル）
```bash
Tanuki_pretraining$ ./training/one_RTX3090_test.sh　　　##実行場所に注意
```

## 本番用のA100　初期学習スクリプト
```bash
Tanuki_pretraining$ ./training/8B-A100_4_pretraining_initial_v2.sh　　　##実行場所に注意
```

## 本番用のA100　継続事前学習スクリプト
```bash
Tanuki_pretraining$ ./training/8B-A100_4_pretraining_continue_v2.sh　　　##実行場所に注意
```


## megatronの引数について

### モデルの構造と並列化
```
- **`-tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}`:** テンソル並列化のサイズ
- **`-pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}`:** パイプライン並列化のサイズ
- **`-context-parallel-size ${CONTEXT_PARALLEL_SIZE}`:** コンテキスト並列化のサイズ
- **`-sequence-parallel`:** シーケンス並列化の有無
- **`-num-layers ${NUM_LAYERS}`:** Transformerモデルの層数
- **`-hidden-size ${HIDDEN_SIZE}`:** モデルの隠れ層の次元数
- **`-ffn-hidden-size ${FFN_HIDDEN_SIZE}`:** Feed-Forward Network の隠れ層の次元数
- **`-num-attention-heads ${NUM_HEADS}`:** Attentionheadsの数
- **`-group-query-attention`:** グループ化されたクエリアテンションを有無
- **`-num-query-groups ${NUM_KEY_VALUE_HEADS}`:** クエリグループの数
```
### 学習設定
```
- **`-seq-length ${SEQ_LENGTH}`:** 入力シーケンスの最大長
- **`-max-position-embeddings ${SEQ_LENGTH}`:** 位置エンコーディングの最大長
- **`-micro-batch-size ${MICRO_BATCH_SIZE}`:** マイクロバッチサイズ
- **`-global-batch-size ${GLOBAL_BATCH_SIZE}`:** グローバルバッチサイズ
- **`-train-iters ${TRAIN_STEPS}`:** 学習ステップ数
- **`-tokenizer-type SentencePieceTokenizer`:** トークナイザーとして SentencePiece の有無
- **`-tokenizer-model ${TOKENIZER_MODEL}`:** トークナイザーモデルのパス
- **`-data-path ${TRAIN_DATA_PATH}`:** 学習データのパス
- **`-split 998,1,1`:** データを訓練データ、検証データ、テストデータに分割する割合
- **`-distributed-backend nccl`:** 分散処理に NCCL を使用有無
- **`-init-method-std 0.008`:** 初期化方法の標準偏差
- **`-lr ${LR}`:** 学習率
- **`-min-lr ${MIN_LR}`:** 最小学習率
- **`-lr-decay-style cosine`:** 学習率の減衰方法としてコサイン減衰
- **`-lr-decay-iters ${LR_DECAY_ITERS}`:** 学習率を減衰させる総ステップ数
- **`-weight-decay ${WEIGHT_DECAY}`:** 重み減衰
- **`-clip-grad ${GRAD_CLIP}`:** 勾配クリッピングの閾値
- **`-lr-warmup-iters ${LR_WARMUP_STEPS}`:** 学習率のウォームアップステップ数
- **`-optimizer adam`:** optimizerの設定としてAdam
- **`-adam-beta1 0.9`:** Adam オプティマイザのベータ1パラメータ
- **`-adam-beta2 0.95`:** Adam オプティマイザのベータ2パラメータ
- **`-adam-eps 1e-05`:** Adam オプティマイザのエプシロン値
- **`-log-interval 1`:** ログ出力のインターバル
- **`-save-interval 500`:** チェックポイントを保存するインターバル
- **`-eval-interval 10000`:** 評価を行うインターバル
- **`-eval-iters 10`:** 評価時に使用するバッチ数
```
### モデルの最適化とハードウェア
```
- **`-bf16`:** BF16
- **`-use-checkpoint-args`:** チェックポイント
- **`-untie-embeddings-and-output-weights`:** エンベディングと出力重み
- **`-no-position-embedding`:** 位置エンコーディング
- **`-position-embedding-type rope`:** 位置エンコーディングとして RoPE (Rotary Position Embeddings)
- **`-rope-theta 500000.0`:** RoPE のパラメータ
- **`-disable-bias-linear`:** 線形層のバイアスの未使用
- **`-use-mcore-models`:** MCore モデル使用
- **`-normalization RMSNorm`:** 正規化として RMSNorm を用
- **`-norm-epsilon 1e-5`:** RMSNorm のイプシロン値
- **`-no-masked-softmax-fusion`:** マスク付きソフトマックスの融合無効
- **`-attention-dropout 0.0`:** アテンション層のドロップアウト率
- **`-hidden-dropout 0.0`:** 隠れ層のドロップアウト率を
- **`-swiglu`:** SwiGLU 活性化関数
- **`-use-flash-attn`:** Flash Attention
- **`-attention-softmax-in-fp32`:** アテンションのソフトマックス計算を FP32
- **`-recompute-activations`:** 活性化関数の再計算を有効
- **`-recompute-granularity "selective"`:** 再計算の粒度を
- **`-transformer-impl "transformer_engine"`:** Transformer エンジンの実装を使用すること
- **`-fp8-format 'hybrid'`:** FP8 フォーマット
- **`-fp8-amax-compute-algo max`:** FP8 の最大値計算アルゴリズム
- **`-fp8-amax-history-len 1024`:** FP8 の最大値計算履歴の長さ
- **`-use-z-loss`:** Z-Loss
- **`-log-throughput`:** スループットのログ出力
```
### 実験管理
```
- **`-wandb-name ${JOB_NAME}`:** Weights & Biases のジョブ名
- **`-wandb-project "Llama-3-8B"`:** Weights & Biases のプロジェクト名
- **`-wandb-entity "weblab-geniac1"`:** Weights & Biases のエンティティ名
```
