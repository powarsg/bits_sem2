# Dataset Card — News Summary Dataset (Inshorts News Data)

**Name:** News Summary Dataset (a.k.a. "Inshorts News Data")

**Source:** Kaggle — https://www.kaggle.com/datasets/sunnysai12345/news-summary
Mirrored (raw CSV files, used directly in this project) at:
https://github.com/sunnysai12345/News_Summary

**License:** The dataset author has not published an explicit open-data
license (e.g. CC-BY, CC0, ODbL) alongside the files. The data is shared
publicly on Kaggle and GitHub for research and educational use. It is used
here strictly for non-commercial academic coursework, as explicitly
permitted by the assignment brief ("groups may also scrape openly
licensed RSS feeds" / "easily available public dataset").

**Composition**

| File | Records | Columns used | Description |
|---|---|---|---|
| `news_summary_more.csv` | 98,401 | `headlines`, `text` | Short article ("Inshort", ~50-66 words) + headline pairs, scraped from the Inshorts app. **Primary training corpus.** |
| `news_summary.csv` | 4,514 (4,396 usable) | `headlines`, `ctext` | Full-length original article text (median 283 words) + headline, scraped from source publications (The Hindu, Indian Times, The Guardian, etc.). Used only to source realistic longer demo articles for the application (`sample_inputs/`), not for training. |

**Size used:** 98,373 article–headline pairs after cleaning (within the
20,000–200,000 record range specified by the assignment). A 6,000 /
800 / 800 train/val/test **subset** was sampled for the model actually
trained in this submission, to keep fine-tuning time reasonable on
CPU-only hardware (explicitly permitted: "a subset may be sampled to
keep training time reasonable"). The full processed split
(88,535 / 4,918 / 4,920) is produced by `prepare_data.py` and the same
code trains on it directly given GPU access.

**Text statistics (full cleaned corpus, n = 98,373)**

| | mean (words) | median (words) |
|---|---|---|
| Article | 58.2 | 59 |
| Headline | 9.6 | 10 |

Article length sits at the lower end of the assignment's "50–400 word"
guidance (Inshorts articles are themselves short-form ~60-word news
capsules by design) rather than spanning the full range; this is noted as
a limitation in the project report. The supplementary `news_summary.csv`
full articles (median 283 words) show the pipeline also handles inputs at
the upper end of the target range.

**Fields available but not used:** `author`, `date`, `read_more` (URL) in
`news_summary.csv` — could support category/date filtering in future
work, but no explicit `category` field is present in either file.
