# integrated-text-summarizer-sentiment
An end-to-end NLP pipeline that generates abstractive summaries with BART and classifies sentiment with DistilBERT, served through a Streamlit UI. (B.Tech CSE-AIML).
# Integrated Text Summarization and Sentiment Analysis
An end-to-end NLP pipeline that:
1. Takes raw text (news article, review, abstract, etc.)
2. Generates an **abstractive summary** using **BART** (`facebook/bart-large-cnn`)
3. Runs **sentiment classification** on that summary using **DistilBERT**
   (`distilbert-base-uncased-finetuned-sst-2-english`)
4. Returns both as a combined response, rendered through a Streamlit UI

## Project structure

```
.
├── app.py              # Streamlit frontend
├── src/
│   └── pipeline.py     # Core summarize_and_analyze() function
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Or use the pipeline directly in Python:

```python
from src.pipeline import summarize_and_analyze

result = summarize_and_analyze("Your article text here...")
print(result)
# {"summary": "...", "sentiment": "POSITIVE", "confidence": 0.9987}
```

## Approach

- **Input handling:** text is truncated to a 1024-token budget before summarization.
- **Summarization:** BART with `num_beams=4`, `length_penalty=2.0`,
  `max_length=150`, `min_length=30` — chosen for factual grounding over
  purely extractive methods.
- **Sentiment:** DistilBERT runs on the *generated summary* rather than
  the raw article, so the sentiment reflects the condensed, editorial
  version of the text.

## Complexity

- Summarization (encoder self-attention across all layers): **O(L·n²·d)**
  for input length *n*, hidden size *d*, *L* layers; beam decoding adds
  roughly **O(b·m·d²)** for beam width *b* and generated length *m*.
- Sentiment classification: **O(L'·m²·d')** — cheap, since it only runs
  once on the short summary (≤150 tokens), not the full article.
- Overall the pipeline is dominated by the **O(n²)** self-attention cost
  on the input article, which is why input length is capped at 1024 tokens.

## Tech stack

- Python 3.10, PyTorch, HuggingFace Transformers, Streamlit
- Datasets: CNN/Daily Mail (summarization), SST-2 (sentiment fine-tuning)
