# SentiScope

A Streamlit sentiment-analysis web app. The model trains **once at server startup** from a fixed CSV — visitors open the site, see a live statistics dashboard, and can analyse any review instantly.

## Project Structure

```
sentiscope/
├── app.py                        # Streamlit entry point
├── config.py                     # DATA_PATH, SAMPLE_SIZE — edit this
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Theme & server settings
├── data/
│   └── Reviews.csv               # ← drop your dataset here
└── src/
    ├── preprocessing/
    │   ├── cleaner.py            # Text cleaning & NLTK setup
    │   └── dataset.py            # CSV loading, labelling, stats
    ├── models/
    │   └── trainer.py            # TF-IDF, NB & LR training, inference
    └── ui/
        ├── styles.py             # Custom CSS
        └── components.py         # Hero, stats dashboard, model cards, analyser
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Drop your Reviews.csv into data/
cp /path/to/Reviews.csv data/

# 3. (Optional) edit config.py to change DATA_PATH or SAMPLE_SIZE

# 4. Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).  
The server trains once on startup — every visitor shares the same trained model.

## Expected CSV Format

| Column  | Description              |
|---------|--------------------------|
| `Score` | Integer star rating (1–5)|
| `Text`  | Review body text         |

Scores 1–2 → **negative**, 4–5 → **positive**, 3 → excluded.

Compatible with the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle.

## Configuration (`config.py`)

| Setting        | Default              | Description                        |
|----------------|----------------------|------------------------------------|
| `DATA_PATH`    | `data/Reviews.csv`   | Path to the CSV file               |
| `SAMPLE_SIZE`  | `50_000`             | Rows sampled for training          |
| `RANDOM_STATE` | `42`                 | Reproducibility seed               |
