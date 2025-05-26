## データ合成

  - [Iterative DPO, LLM-as-a-Judge](synthetic_data/iterative-dpo)
    - 合成したプリファレンスデータセット (DPO/RLHF用)
      - [synth-dpo-basic-reasoning-nemotron-4-raw](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-reasoning-nemotron-4-raw)
      - [synth-dpo-basic-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-reasoning-nemotron-4)
      - [synth-dpo-basic-math-nemotron-4-raw](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-math-nemotron-4-raw)
      - [synth-dpo-basic-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-dpo-basic-math-nemotron-4)
  - [Topic-Hub, Persona-Hub](synthetic_data/topic-hub)
    - 合成したSFTデータセット
      - [synth-topic-jp-basic-math-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-basic-math-calm3)
      - [synth-topic-jp-basic-reasoning-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-basic-reasoning-calm3)
      - [synth-topic-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-reasoning-nemotron-4)
      - [synth-topic-jp-reasoning-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-reasoning-calm3)
      - [synth-topic-jp-highschoolmath-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-highschoolmath-nemotron-4)
      - [synth-persona-jp-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-math-nemotron-4)
      - [synth-persona-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-persona-jp-reasoning-nemotron-4)
      - [synth-topic-jp-coding-calm3](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-topic-jp-coding-calm3)
  - [Magpie](synthetic_data/magpie)
    - 合成したSFTデータセット
      - [synth-magpie-jp-math-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-math-nemotron-4)
      - [synth-magpie-jp-coding-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-coding-nemotron-4)
      - [synth-magpie-jp-reasoning-nemotron-4](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-magpie-jp-reasoning-nemotron-4)
  - [ルールベース](synthetic_data/rule-based)
    - 合成した事前学習データセット
      - [synth-rule-arithmetic-qa](https://huggingface.co/datasets/team-hatakeyama-phase2/synth-rule-arithmetic-qa)
- データに対するスコア付け
  - [Ask-LLM](synthetic_data/ask-llm)
  - [言語判定](synthetic_data/lang-identifier)
- Nemotron-4 推論環境構築
  - [vLLM FP8 量子化](synthetic_data/nemotron-vllm-fp8)
  - [Megatron](synthetic_data/nemotron-megatron)

太田 晋, “FinePersonas を用いた日本語指示データの合成”, NLP2025 併設ワークショップ JLR2025, 2025-03-14, https://jedworkshop.github.io/JLR2025/materials/a-2.pdf
@inproceedings{ota2025finepersonas,
  author    = {太田 晋},
  title     = {FinePersonasを用いた日本語指示データの合成},
  booktitle = {NLP2025 併設ワークショップ JLR2025},
  year      = {2025},
  month     = mar,
  day       = {14},
}