# ml-jitumu-code — 『Pythonで解く 業務課題集』サンプルコード

本書の各章に載せたコードを、**検証単位ごとに 1 つの Notebook** にまとめたもの。
フォルダ番号は**本書の章番号**と一致している（第 1 章＝`01_credit`）。

## 使い方

```bash
git clone https://github.com/ChisouPublishing/ml-jitumu-code.git
cd ml-jitumu-code
pip install -r requirements.txt
```

- **どのノートも単体で完結する。** 他のノートを先に実行する必要はない。
- 各ノートは課題フォルダの **`input/*.csv` を読み込む**（公開データの取得に失敗したときの合成代替と、社内実務で公開データを使えない課題の合成データ）。
  `input/` が無いときはノート内蔵の生成関数へ自動で切り替わる。
- 各ノートの先頭に **出典（章・節）／このノートのパス／必要なパッケージ／実行時間の目安** がある。
- 本書のデータはすべて合成または公開データであり、実務の根拠・実審査基準ではない。

## Inputデータの再生成

`input/*.csv` は `make_inputs.py` で作ったものである。消した・変えたときは作り直す。

```bash
python3 make_inputs.py
```

生成ロジックは各ノートの内蔵関数と同一（同一シード）のため、CSV経由と内蔵生成で同一のデータになる。

## 実行結果の図解

比較チャート（PR曲線・閾値曲線・リフト・利益・残差・スコア分布・ドリフト等）は、各ノートが `matplotlib` で課題フォルダの `outputs/code-result-*.png` に保存する。保存先フォルダは自動作成する。書籍制作環境の `../../../books/python-machine-jitumu/assets/images` が存在する場合は、従来どおりそこへ保存する。ラベルは英語のみ（実行環境のフォントに依存しないため）。

```bash
pip install -r requirements.txt     # まとめて入れる場合
jupyter lab                         # あるいは VS Code / Colab で開く
```

## 一括実行による検証

このフォルダをカレントディレクトリにして実行する。

```bash
zsh run_all_notebooks.sh . "$(command -v python3)" 1800
```

対象環境に `nbconvert` と `ipykernel` が必要。`tmp/` とNotebookチェックポイントは実行対象に含めない。

## 章の一覧

### `01_credit/` — 第 1 章 与信可否審査

| ファイル | 内容 |
|---|---|
| `01_baseline.ipynb` | German Credit（UCI公開データ）の読込・層化分割・ダミー／線形・費用による期待損失（全件承認・全件否決を物差しに） |
| `02_improved.ipynb` | 学習データ内のCVによるOptuna探索・較正・OOFでの閾値と3段振り分け・テストでの最終確認・順列重要度と理由素・公平性（機微な属性の除外と属性別承認率） |
| `input/applications.csv` | 公開データ取得失敗時の合成代替（German Credit形式） |

### `02_fraud/` — 第 2 章 不正検知

| ファイル | 内容 |
|---|---|
| `01_baseline.ipynb` | 不均衡分類のベースラインと比較 |
| `02_improved.ipynb` | 教師あり（HGB）と異常検知の2系統・手口別の再現率（テスト＝既知のholdout＋手口B）・旗上げ率の対応表 |
| `input/transactions.csv`・`input/transactions_modus.csv` | 合成の取引データ（各20000件） |

### `03_targeting/` — 第 3 章 ターゲティング

| ファイル | 内容 |
|---|---|
| `01_baseline.ipynb` | リーク実演（duration）・分類比較・リフト測定（既定はOpenML bank-marketing公開データ、duration を除外） |
| `02_improved.ipynb` | HGB・深さ別の期待利益（成約粗利−接触原価。既定はOpenML bank-marketing公開データ） |
| `input/bank.csv` | 公開データ取得失敗時の合成代替（bank形式・45211件） |

### `04_demand/` — 第 4 章 需要予測

| ファイル | 内容 |
|---|---|
| `01_baseline.ipynb` | 特徴量作成とベースライン比較（既定は月次航空旅客数の公開データ） |
| `02_improved.ipynb` | 分割の比較と残差点検（既定は公開データ） |
| `input/passengers.csv` | 公開データ取得失敗時の合成代替（144か月） |

### `05_inspection/` — 第 5 章 故障予知と外観検査

| ファイル | 内容 |
|---|---|
| `01_baseline.ipynb` | PCAの再構成誤差による良品学習と固定閾値 |
| `02_improved.ipynb` | 調整用の正常による閾値調整（トレードオフ表）・公開画像（digits）での良品学習・Steel Plates Faults（UCI）の不良種別の診断 |
| `input/sensors.csv` | 合成のセンサーデータ（5200件） |

### `06_ops/` — 第 6 章 運用・監視・再学習

| ファイル | 内容 |
|---|---|
| `01_baseline.ipynb` | Pipelineごとの保存・読込・速度測定（まとめて処理／1件ずつ） |
| `02_improved.ipynb` | 二段の監視：PSI（早めの警報）と、学習に使っていないholdoutを基準にした帯別貸倒率（確定後の発火） |
| `input/mini_applications.csv`・`input/ops_base.csv`・`input/ops_new.csv` | 合成の運用データ |

### `07_production/` — 第 7 章 実務への載せ方

| ファイル | 内容 |
|---|---|
| `01_shadow.ipynb` | PoC（過去データのholdout）での比較 → Shadow（新着）での並行記録・不一致の内訳・切替判断 |
| `input/shadow_base.csv`・`input/shadow_new.csv` | 合成の新旧比較データ |

## ローカルの生成物

`credit_pipe.joblib` と `candidate_pipe.joblib` は、それぞれ第6章・第7章のNotebook実行時に学習して保存する。配布済みのモデルは不要で、実行時に再生成できる。
保存したモデル、図の出力、キャッシュ、ログ、内部資料用の `tmp/` はGit対象から除外する。教材用の `input/*.csv` と、その生成スクリプト `make_inputs.py` は公開対象に含める。
