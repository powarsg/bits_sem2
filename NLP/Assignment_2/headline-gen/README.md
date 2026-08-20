# Group 116 — Automatic News Headline Generation

NLP Assignment 2 (S2-25_AIMLCZG530) — end-to-end Encoder–Decoder application
that reads a news article and generates a single-line abstractive headline.

## 1. Project layout

```
headline-gen/
├── data/
│   ├── raw/                      # downloaded source CSVs
│   ├── download_data.sh           # downloads source CSVs into data/raw/
│   ├── prepare_data.py           # cleaning, normalisation, vocab, splits
│   ├── make_train_subset.py      # samples a smaller train/val/test set
│   └── processed/                # cleaned full splits + subset/ + vocab.json + stats.json
├── training/
│   ├── vocab.py                  # word-level tokenizer & vocabulary
│   ├── model.py                  # from-scratch Transformer encoder-decoder
│   ├── dataset.py                # PyTorch Dataset / collate
│   ├── train.py                  # training loop, loss curve, checkpoints
│   └── infer.py                  # greedy + beam-search decoding
├── evaluation/
│   ├── evaluate.py                # ROUGE-1/2/L on the test split
│   └── results/                   # ROUGE scores, evaluation config, sample predictions
├── app/
│   ├── app.py                     # Flask web application
│   ├── templates/index.html
│   └── static/style.css
├── models/transformer_scratch/    # best_model.pt, vocab.json, train_config.json, loss_history.json, loss_curve.png
├── sample_inputs/                 # example .txt / .csv files for testing the app
└── requirements.txt
```

## 2. Dataset

**Name:** News Summary Dataset (a.k.a. "Inshorts News Data")
**Source:** https://www.kaggle.com/datasets/sunnysai12345/news-summary
(mirrored at https://github.com/sunnysai12345/News_Summary)
**License:** No explicit open-data license file is published by the dataset
author; the data is shared publicly on Kaggle for research/educational use.
Used here strictly for non-commercial academic coursework, as permitted by
the assignment brief.
**Size used:** 98,401 raw article–headline pairs (`news_summary_more.csv`);
98,373 remain after cleaning. A subset of 6,000 train / 800 validation /
800 test records was sampled for the actually-trained model in this
submission, to keep training time reasonable on CPU-only hardware (see
§4). The full 88,535/4,918/4,920 processed split is also produced by
`prepare_data.py` and the same training code scales to it directly given a
GPU.
A handful of longer, real full-length articles (`news_summary.csv`,
`ctext` column) are kept separately as realistic demo inputs for the
application (`data/processed/demo_full_articles.csv`, copied into
`sample_inputs/`).

## 3. Local setup

**Python version:** 3.11 is recommended (Python 3.9+ should work). Run every
command below from the `headline-gen/` directory unless a command says
otherwise.

```bash
cd NLP/Assignment_2/headline-gen
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install the remaining dependencies first,
then install PyTorch from its CPU wheel index:

```bash
python -m pip install pandas numpy matplotlib Flask rouge_score nltk beautifulsoup4 lxml
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify that the application dependencies and submitted model can be loaded:

```bash
python -c "from app.app import app; print(app.test_client().get('/health').get_json())"
```

Expected output includes `"status": "ok"`. The first load can take a few
seconds because PyTorch reads the saved checkpoint.

## 4. Reproducing the pipeline

1. Download the two source CSV files into `data/raw/` (network access required).
This step is only required to rebuild the processed data or retrain the model;
the supplied checkpoint can be used directly by the Flask app.

```bash
bash data/download_data.sh
```

2. Clean the dataset and produce deterministic full train/validation/test
splits. This also writes `data/processed/vocab.json`, `stats.json`, and longer
demo articles.

```bash
python data/prepare_data.py
```

3. Create the CPU-friendly training subset:

```bash
python data/make_train_subset.py --n_train 6000 --n_val 800 --n_test 800
```

4. Train the from-scratch Transformer. On CPU this can take a long time; the
saved checkpoint in `models/transformer_scratch/` lets the application run
without retraining.

```bash
python training/train.py \
  --train_csv data/processed/subset/train.csv \
  --val_csv   data/processed/subset/val.csv \
  --out_dir   models/transformer_scratch \
  --epochs 12 --batch_size 24 --vocab_size 8000
```

This writes `best_model.pt`, `vocab.json`, `train_config.json`,
`loss_history.json` and `loss_curve.png` to `models/transformer_scratch/`.
`train_config.json` records both requested `epochs` and `completed_epochs`;
check `training_complete` before treating a run as complete.

To train on the **full** dataset instead (recommended on a GPU machine,
e.g. the BITS OSHA Virtual Lab or Google Colab), simply point at the full
processed splits:

```bash
python training/train.py --train_csv data/processed/train.csv --val_csv data/processed/val.csv \
  --out_dir models/transformer_full --epochs 15 --batch_size 64
```

## 5. Evaluation

```bash
python evaluation/evaluate.py --model_dir models/transformer_scratch \
  --test_csv data/processed/subset/test.csv
```

Writes `evaluation/results/rouge_scores.json` and
`evaluation/results/sample_predictions.csv`, plus
`evaluation/results/evaluation_config.json`. By default, evaluation covers the
complete held-out test split using greedy decoding. To do a quick smoke test
only, add `--n_eval 10`. The evaluation configuration records the model path,
test CSV, requested and completed record count, and decoding method.

## 6. Running the application

```bash
python app/app.py
```

Open **http://127.0.0.1:5000** in a browser. Paste an article, or upload a
`.txt`/`.csv` file (sample files are in `sample_inputs/`), choose beam
size / top-k, and click "Generate Headline(s)". CSV files need an `article`,
`text`, `body`, or `content` column; the first column is used otherwise. The
app accepts files up to 10 MB and processes at most 25 articles per request.
For `.txt` uploads, blank lines separate individual articles.

Quick browser-free smoke test:

```bash
python3 -c "from app.app import app; r = app.test_client().post('/generate', data={'article_text': 'City council approves a public transport plan after a Tuesday vote.', 'beam_size': '1', 'top_k': '1'}); print(r.status_code)"
```

To point the app at a different trained checkpoint:

```bash
MODEL_DIR=models/transformer_full python app/app.py
```

To use a different local port:

```bash
PORT=5050 python app/app.py
```

The health endpoint at `http://127.0.0.1:5000/health` reports whether the
checkpoint can be loaded.

## 7. Known issues / limitations

- The supplied model was trained for **12 epochs** on a 6,000-article subset
  to fit CPU-only training constraints. Training completed in approximately
  9.1 minutes, with final validation cross-entropy loss of 5.7468. Headline
  quality, fluency, and factual precision should improve with full-dataset,
  GPU-based training (see §4).
- Word-level vocabulary (8,000 tokens) means rare words/out-of-vocabulary
  named entities are mapped to `<unk>`; a larger vocabulary or sub-word
  tokenizer (e.g. BPE) would reduce this.
- The Flask app loads the model once per process (`MODEL_DIR` env var
  selects the checkpoint) and is intended for local demonstration, not
  production traffic.

## 8. Group details

| Name | BITS ID | Contribution % |
|---|---|---|
| SUJEET KUMAR YADAV | 2025AA05326 | 20% |
| SHELAR SACHIN KRISHNA | 2025AA05387 | 20% |
| POWAR SAGAR GANPATI | 2025AA05421 | 20% |
| PRASAD SHIVAJI KULKARNI | 2025AA05444 | 20% |
| PATTIPATI SAI CHANDAN SINGH | 2025AB05149 | 20% |

Group 116, Natural Language Processing (S2-25_AIMLCZG530).
