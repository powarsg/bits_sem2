#!/usr/bin/env bash
# Downloads the raw dataset CSVs into data/raw/
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw
curl -sL -o raw/news_summary.csv \
  https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary.csv
curl -sL -o raw/news_summary_more.csv \
  https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv
echo "Downloaded to data/raw/"
