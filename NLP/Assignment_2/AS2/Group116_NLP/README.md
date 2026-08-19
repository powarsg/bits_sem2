# Group 116 — Automatic News Headline Generation

NLP Assignment 2 (S2-25_AIMLCZG530) — end-to-end Encoder–Decoder application
that reads a news article and generates a single-line abstractive headline.

## 1. Project layout

```
headline-gen/
├── data/
│   ├── raw/                      # downloaded source CSVs
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
│   └── results/                   # rouge_scores.json, sample_predictions.csv
├── app/
│   ├── app.py                     # Flask web application
│   ├── templates/index.html
│   └── static/style.css
├── models/transformer_scratch/    # best_model.pt, vocab.json, train_config.json, loss_history.json, loss_curve.png
├── sample_inputs/                 # example .txt / .csv files for testing the app
├── report/                        # Group116.pdf and screenshots
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

**Python version:** 3.11 (3.9+ should work)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
# CPU-only PyTorch (smaller download) — optional alternative to the line above:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Reproducing the pipeline

```bash
cd data
python3 prepare_data.py                 # cleans full dataset -> data/processed/{train,val,test}.csv
python3 make_train_subset.py --n_train 6000 --n_val 800 --n_test 800

cd ../training
python3 train.py \
  --train_csv ../data/processed/subset/train.csv \
  --val_csv   ../data/processed/subset/val.csv \
  --out_dir   ../models/transformer_scratch \
  --epochs 12 --batch_size 24 --vocab_size 8000
```

This writes `best_model.pt`, `vocab.json`, `train_config.json`,
`loss_history.json` and `loss_curve.png` to `models/transformer_scratch/`.

To train on the **full** dataset instead (recommended on a GPU machine,
e.g. the BITS OSHA Virtual Lab or Google Colab), simply point at the full
processed splits:

```bash
python3 train.py --train_csv ../data/processed/train.csv --val_csv ../data/processed/val.csv \
  --out_dir ../models/transformer_full --epochs 15 --batch_size 64
```

## 5. Evaluation

```bash
cd evaluation
python3 evaluate.py --model_dir ../models/transformer_scratch \
  --test_csv ../data/processed/subset/test.csv
```

Writes `evaluation/results/rouge_scores.json` and
`evaluation/results/sample_predictions.csv`.

## 6. Running the application

```bash
cd app
python3 app.py
```

Open **http://127.0.0.1:5000** in a browser. Paste an article, or upload a
`.txt`/`.csv` file (sample files are in `sample_inputs/`), choose beam
size / top-k, and click "Generate Headline(s)".

To point the app at a different trained checkpoint:

```bash
MODEL_DIR=../models/transformer_full python3 app.py
```

## 7. Known issues / limitations

- The model shipped in this submission was trained on a **6,000-article
  subset** for a small number of epochs to fit CPU-only training time
  constraints; headline quality (fluency, factual precision) will improve
  substantially with full-dataset / GPU training (see §4).
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
