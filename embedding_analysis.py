from collections import Counter
from pathlib import Path
import json
import re
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
import nltk
from utils import BasicTokenizer, PretrainedEmbeddings, download_pretrained, cos_similarity

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "embedding_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_PAIRS = [
    ("woman", "women", "morphological variants"),
    ("immigrant", "immigrants", "morphological variants"),
    ("migrant", "refugee", "related immigration terms"),
    ("hate", "hatred", "related negative concepts"),
    ("love", "hate", "opposing sentiment"),
    ("good", "bad", "opposing sentiment"),
    ("illegal", "immigrant", "co-occurrence/associations"),
    ("not", "hate", "negation vocabulary"),
]

# dataset specific, TweetEval, hate subset
DATASET_NAME = "cardiffnlp/tweet_eval"
LABEL_NAMES = ["non-hate", "hate"]
NUM_CLASSES = len(LABEL_NAMES)


def retrieve_dataset() -> tuple:
    # load ds from HF
    ds = load_dataset("cardiffnlp/tweet_eval", "hate")

    # train: 9000, validation: 1000, test:2970
    X_train = ds['train']['text']
    y_train = ds['train']['label']
    X_validation = ds['validation']['text']
    y_validation = ds['validation']['label']
    X_test = ds['test']['text']
    y_test = ds['test']['label']

    return {"train": (X_train, y_train),
            "validation": (X_validation, y_validation),
            "test": (X_test, y_test)}


def tokenize(text, embedding_model):
    tokenizer = BasicTokenizer(lowercase=True,
                               remove_punct=False,
                               remove_stopwords=False, 
                               language='english',
                               vocab=embedding_model.vocab)

    return tokenizer.tokens(text)


def fasttext_embed():
    # the line below corresponds to tuple ('wiki-news-300d-1M.vec.zip', 300, True)
    embtable_version, embtable_dims, embtable_metadata = download_pretrained('fasttext')
    trained_emb_model = PretrainedEmbeddings(embtable_version, embtable_dims, skip_first_line=embtable_metadata)

    return trained_emb_model


def analyze_coverage(dataset, embedding_model, labels):
    rows = []
    unknown = Counter()
    all_tokens = Counter()

    for split in ("train", "validation", "test"):
        texts, labels_split = dataset[split]

        for text, label in zip(texts, labels_split):
            tokens = tokenize(text, embedding_model)
            # tokenizer.ids() maps every unknown to <unk>, hiding unk_words status.
            unk_words = [token for token in tokens if token not in embedding_model.vocab]
            unknown.update(unk_words)
            all_tokens.update(tokens)
            rows.append({
                "split": split,
                "label": int(label),
                "label_name": labels[int(label)],
                "text": text,
                "num_tokens": len(tokens),
                "unk_words_tokens": len(unk_words),
                "covered_tokens": len(tokens) - len(unk_words),
                "token_coverage": ((len(tokens) - len(unk_words)) / len(tokens)
                                   if tokens else np.nan),
                "contains_unk_words": bool(unk_words),
                "has_hashtag": bool(re.search(r"#\w+", text)),
                "has_mention": bool(re.search(r"@\w+", text)),
                "unk_words_examples": ", ".join(unk_words[:8]),
            })
    detail = pd.DataFrame(rows)

    distinct_tokens = set(all_tokens)
    overall = {"embedding_vocabulary_size_including_special_tokens": len(embedding_model.vocab),
               "distinct_tweeteval_tokens": len(distinct_tokens),
               "overall_token_coverage": (len(tokens) - len(unk_words)) / len(tokens)}
    
    most_frequent_unks = pd.DataFrame(unknown.most_common(100), columns=["token", "count"])

    return overall, detail, most_frequent_unks


def analyze_semantics(embedding_model, word_pairs):
    results = []
    for word_a, word_b, category in word_pairs:
        present_a = word_a in embedding_model
        present_b = word_b in embedding_model
        # Only score when both words genuinely have vectors. Otherwise the
        # unknown-token mean vector would produce misleading similarities.
        if present_a and present_b:
            score = cos_similarity(embedding_model.embed_word(word_a),
                                   embedding_model.embed_word(word_b))
        else:
            score = np.nan
        results.append({"word_a": word_a,
                        "word_b": word_b,
                        "a_in_vocabulary": present_a,
                        "b_in_vocabulary": present_b,
                        "cosine_similarity": score})

    return pd.DataFrame(results)


def main():
    start_time = time.time()

    print("Loading TweetEval hate dataset...")
    hate_dataset = retrieve_dataset()

    print('Load FastText embedding model...')
    fasttext_emb = fasttext_embed()
    
    print('Coverage analysis started...')
    overall, detail, most_frequent_unks = analyze_coverage(hate_dataset, fasttext_emb, LABEL_NAMES)
    
    print('Similarity analysis started...')
    similarities = analyze_semantics(fasttext_emb, WORD_PAIRS)

    detail.to_csv(OUTPUT_DIR / "tweet_coverage.csv", index=False)
    similarities.to_csv(OUTPUT_DIR / "semantic_similarities.csv", index=False)
    most_frequent_unks.to_csv(OUTPUT_DIR / "most_frequently_unknown.csv", index=False)
    with (OUTPUT_DIR / "overall_statistics.json").open("w") as f:
        json.dump(overall, f, indent=2)
    print('Embedding model analysis saved to results/embedding_analysis/')


if __name__ == "__main__":
    main()
