"""Requirements 4–5: paired, repeated-stratified CV comparing LSTM and embed-and-pool.

Place beside embed_pool_experiment_fixed.py, lstm_fixed.py, embedding_analysis.py,
and utils.py. Uses only the TweetEval training split for cross-validation.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from embedding_analysis import retrieve_dataset, fasttext_embed
from utils import BasicTokenizer
from embed_pool_experiment import (
    PoolClassifier, compact_embeddings, make_examples,
    collate_batch, run_epoch, set_seed,
)
from models.lstm import LSTMEncoder

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results' / 'recurrent_experiments'

# Start with only a few targeted contrasts; resist a huge factorial search.
CONFIGS = {
    'pool_mean':     dict(encoder='pool', hidden_size=0, word_dropout=0., trainable=False, fraction=1.),
    'lstm_20':       dict(encoder='lstm', hidden_size=20, word_dropout=0., trainable=False, fraction=1.),
    'lstm_50':       dict(encoder='lstm', hidden_size=50, word_dropout=0., trainable=False, fraction=1.),
    'lstm_dropout':  dict(encoder='lstm', hidden_size=20, word_dropout=0.2, trainable=False, fraction=1.),
    'lstm_finetune': dict(encoder='lstm', hidden_size=20, word_dropout=0., trainable=True, fraction=1.),
    'lstm_halfdata': dict(encoder='lstm', hidden_size=20, word_dropout=0., trainable=False, fraction=0.5),
}


class TextClassifier(nn.Module):
    """Wrap existing LSTMEncoder in the same linear head as PoolClassifier."""
    def __init__(self, matrix, pad_idx, config):
        super().__init__()
        self.encoder = LSTMEncoder(
            hidden_size=config['hidden_size'], embedding_matrix=matrix.clone(),
            padding_idx=pad_idx, freeze_embeddings=not config['trainable'],
            word_dropout=config['word_dropout'],
        )
        self.classifier = nn.Linear(self.encoder.enc_dim, 2)

    def forward(self, x):
        return self.classifier(self.encoder(x))


def prepare_examples(texts, labels, tokenizer, unk_idx):
    # Packed recurrent sequences must have at least one non-pad token.
    pairs = make_examples(texts, labels, tokenizer)
    return [(ids if ids else [unk_idx], label) for ids, label in pairs]


def make_model(matrix, pad_idx, config):
    if config['encoder'] == 'pool':
        return PoolClassifier(matrix, pad_idx, 'mean', pretrained=True)
    return TextClassifier(matrix, pad_idx, config)


def one_run(config_name, config, seed, fold, training, validation, matrix,
            pad_idx, device, args):
    set_seed(seed * 1000 + fold)
    collate = lambda b: collate_batch(b, pad_idx)
    # Same ordering/randomness for all configurations on the same fold.
    training_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed * 1000 + fold),
        collate_fn=collate,
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size,
                                   collate_fn=collate)
    model = make_model(matrix, pad_idx, config).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    best_loss, best_epoch, best_state, patience_left = float('inf'), 0, None, args.patience
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, training_loader, device, optimizer)
        val_metrics = run_epoch(model, validation_loader, device)
        history.append(dict(config=config_name, seed=seed, fold=fold, epoch=epoch,
                            **{f'train_{k}': v for k, v in train_metrics.items()},
                            **{f'val_{k}': v for k, v in val_metrics.items()}))
        if val_metrics['loss'] < best_loss - 1e-5:
            best_loss, best_epoch = val_metrics['loss'], epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    model.load_state_dict(best_state)
    selected = run_epoch(model, validation_loader, device)
    result = dict(config=config_name, seed=seed, fold=fold, best_epoch=best_epoch,
                  train_examples=len(training), val_examples=len(validation),
                  **{f'val_{k}': v for k, v in selected.items()})
    del model, optimizer
    if device == 'cuda':
        torch.cuda.empty_cache()
    return result, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--configs', nargs='+', choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    if args.folds < 2:
        parser.error('--folds must be at least 2')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load fastText once. Compact the matrix without using the test split.
    data = retrieve_dataset()
    texts, labels = data['train']
    emb_model = fasttext_embed()
    base_tok = BasicTokenizer(True, False, False, 'english', emb_model.vocab)
    vocab, matrix = compact_embeddings(emb_model, list(texts), base_tok)
    del emb_model
    tokenizer = BasicTokenizer(True, False, False, 'english', vocab)
    examples = prepare_examples(texts, labels, tokenizer, vocab.default_index)
    labels = np.asarray(labels)
    rows, histories = [], []
    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
            validation = [examples[i] for i in val_idx]
            for config_name in args.configs:
                config = CONFIGS[config_name]
                selected_idx = train_idx
                if config['fraction'] < 1.:
                    selected_idx, _ = train_test_split(
                        train_idx, train_size=config['fraction'],
                        stratify=labels[train_idx], random_state=seed * 1000 + fold)
                training = [examples[i] for i in selected_idx]
                print(f'seed={seed} fold={fold + 1}/{args.folds} config={config_name}', flush=True)
                result, hist = one_run(config_name, config, seed, fold, training,
                                       validation, matrix, vocab.index('<pad>'), device, args)
                rows.append(result)
                histories.extend(hist)
                # Save progress even when running a long experiment.
                pd.DataFrame(rows).to_csv(RESULTS_DIR / 'fold_results.csv', index=False)
                pd.DataFrame(histories).to_csv(RESULTS_DIR / 'epoch_history.csv', index=False)
                print(f"  macro_f1={result['val_macro_f1']:.4f}, hate_f1={result['val_hate_f1']:.4f}")

    results = pd.DataFrame(rows)
    summary = results.groupby('config', as_index=False).agg(
        macro_f1_mean=('val_macro_f1','mean'), macro_f1_std=('val_macro_f1','std'),
        hate_f1_mean=('val_hate_f1','mean'), hate_f1_std=('val_hate_f1','std'),
        accuracy_mean=('val_accuracy','mean'), runs=('val_macro_f1','size'),
        mean_best_epoch=('best_epoch','mean'),
    ).sort_values('macro_f1_mean', ascending=False)
    summary.to_csv(RESULTS_DIR / 'summary.csv', index=False)

    # Paired comparisons: same seed/fold, identical validation examples.
    baseline = results[results.config == 'pool_mean'][['seed','fold','val_macro_f1']].rename(
        columns={'val_macro_f1':'pool_macro_f1'})
    if not baseline.empty:
        paired = results.merge(baseline, on=['seed','fold'], how='inner')
        paired['delta_macro_f1_vs_pool'] = paired.val_macro_f1 - paired.pool_macro_f1
        paired.to_csv(RESULTS_DIR / 'paired_vs_pool.csv', index=False)
        paired.groupby('config', as_index=False)['delta_macro_f1_vs_pool'].agg(
            ['mean','std','count']).to_csv(RESULTS_DIR / 'paired_summary.csv')

    with (RESULTS_DIR / 'settings.json').open('w') as handle:
        json.dump(dict(**vars(args), device_used=device, vocab_size=len(vocab),
                       configs={k: CONFIGS[k] for k in args.configs},
                       validation_policy='Stratified K-fold on train split; test unused'),
                  handle, indent=2)
    print('\n' + summary.to_string(index=False))
    print(f'\nOutputs: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
