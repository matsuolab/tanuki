# Tanuki Project 🦝

**日本語での対話・作文性能に力点を置いた大規模言語モデルの開発**

[![Profile views](https://komarev.com/ghpvc/?username=tanuki-llm&style=flat-square&color=brightgreen)](https://tanuki-llm.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](http://tanuki-llm.github.io/3G1-GS-6-04.pdf)

## 📋 概要

**Tanuki** は、GENIACプロジェクトによって開発された日本語特化の大規模言語モデル（LLM）のデファクタリング済みコードを公開しています。継続事前学習、事後学習（ファインチューニング）にお使いください。

> ⚠️ **注意**: 開発中に用いたコードそのものではありません

## 🚀 主な成果

- **Tanuki-8B**: 10B級モデルを上回る性能を達成
- **Tanuki-8×8B**: 国内でフルスクラッチ開発されたモデルとしてトップレベルの性能
- **Japanese MT-Bench (JMT-Bench)**: 2024年8月末時点での評価結果

　🌐 詳細は　プロジェクトページ　をご覧ください。[https://tanuki-llm.github.io/](https://tanuki-llm.github.io/) |

## 📁 リポジトリ構成

### 事前学習用コード
```
/Tanuki_pretraining/README.md
```
詳細な事前学習の手順と設定について記載

### 事後学習用コード  
```
/Tanuki_fine_tuning/sftlab/README.md
```
Supervised Fine-Tuning (SFT) の実行方法

### 合成データ
```
/synthetic_data/README.md  
```
学習に使用した合成データの詳細

## 🔗 関連リンク

| リソース | 説明 | リンク |
|---------|------|--------|
| 🌐 **プロジェクトページ** | 詳細な解説とデモ | [https://tanuki-llm.github.io/](https://tanuki-llm.github.io/) |
| 📄 **論文** | 技術的詳細 | [PDF](http://tanuki-llm.github.io/3G1-GS-6-04.pdf) |
| 🤗 **HuggingFace** | モデルのダウンロード | [weblab-GENIAC](https://huggingface.co/weblab-GENIAC) |
| 📝 **Tech Blog** | 開発ブログ | [Zenn](https://zenn.dev/p/matsuolab) |
| 💬 **お問い合わせ** | 質問・バグ報告 | [Issues](https://github.com/tanuki-llm/tanuki-llm.github.io/issues) |

## ℹ️ プロジェクト背景

この成果は、NEDO（国立研究開発法人新エネルギー・産業技術総合開発機構）の助成事業「ポスト5G情報通信システム基盤強化研究開発事業」（JPNP20017）の結果得られたものです。

## 📄 ライセンス

このプロジェクトはApache 2.0ライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご確認ください。

## 🤝 貢献

プロジェクトへの貢献を歓迎します！バグ報告や機能要望は[Issues](https://github.com/tanuki-llm/tanuki-llm.github.io/issues)までお願いします。
