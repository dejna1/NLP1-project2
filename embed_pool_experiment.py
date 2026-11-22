import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from embedding_analysis import fasttext_embed, retrieve_dataset
from utils import BasicTokenizer, Vocabulary, collate_sequences, EmbedAndPoolEncoder

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results' / 'embed_and_pool'


def make_examples(texts, labels, tokenizer):
    """Prepare (token IDs, label) pairs once using the tutorial tokenizer."""
    return [(tokenizer.ids(text), int(label)) for text, label in zip(texts, labels)]


def collate_batch(batch, pad_idx):
    sequences, labels = zip(*batch)
    x = collate_sequences(sequences, padding_value=pad_idx)
    return x, torch.tensor(labels, dtype=torch.long)


class PoolClassifier(nn.Module):
    """Tutorial-style encoder followed by a trainable softmax regressor."""
    def __init__(self, embedding_matrix, pad_idx, pool, pretrained):
        super().__init__()
        matrix = embedding_matrix.clone() if pretrained else torch.randn_like(embedding_matrix) * 0.1
        matrix[pad_idx] = 0
        self.encoder = EmbedAndPoolEncoder(
            embedding_matrix=matrix, average_pool=(pool == "mean"),
            padding_idx=pad_idx, freeze_embeddings=True,
        )
        d = self.encoder.enc_dim
        self.classifier = nn.Linear(d, 2)

    def forward(self, x):
        return self.classifier(self.encoder(x))  # logits for CrossEntropyLoss


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, predictions, gold = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append((loss.item(), len(y)))
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        gold.extend(y.cpu().tolist())
    return {
        'loss': sum(v * n for v, n in losses) / sum(n for _, n in losses),
        'accuracy': accuracy_score(gold, predictions),
        'micro_f1': f1_score(gold, predictions, average='micro', zero_division=0),
        'macro_f1': f1_score(gold, predictions, average='macro', zero_division=0),
        'hate_f1': f1_score(gold, predictions, labels=[1], average='macro', zero_division=0),
    }


def compact_embeddings(emb_model, texts, tokenizer):
    original = emb_model.vocab
    specials = [emb_model.pad_token, emb_model.unk_token]
    seen = set(specials)
    words = list(specials)
    for text in texts:
        for token in tokenizer.tokens(text):
            if token in original and token not in seen:
                seen.add(token)
                words.append(token)
    vocab = Vocabulary(words, default=emb_model.unk_token, frozen=True)
    matrix = torch.from_numpy(emb_model.vectors[
        [original.index(w) for w in words]
    ].copy())
    return vocab, matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else (
        'cpu' if args.device == 'auto' else args.device
    )
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    data = retrieve_dataset()
    embeddings = fasttext_embed()
    base_tokenizer = BasicTokenizer(True, False, False, 'english', embeddings.vocab)
    # Build compact vocabulary using train/validation text only (not the test set).
    all_texts = list(data['train'][0]) + list(data['validation'][0])
    compact_vocab, matrix = compact_embeddings(embeddings, all_texts, base_tokenizer)
    del embeddings
    tokenizer = BasicTokenizer(True, False, False, 'english', compact_vocab)
    pad_idx = compact_vocab.index('<pad>')
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['validation']
    train_ds = make_examples(train_texts, train_labels, tokenizer)
    val_ds = make_examples(val_texts, val_labels, tokenizer)
    collate = lambda batch: collate_batch(batch, pad_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

    # 4 design choices plus one pretraining ablation under otherwise identical settings.
    configurations = [
        ('mean_linear', 'mean', 'linear', True),
        ('sum_linear', 'sum', 'linear', True),
    ]
    all_history, summary = [], []
    for name, pool, head, pretrained in configurations:
        set_seed(args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate,
                                  generator=torch.Generator().manual_seed(args.seed))
        model = PoolClassifier(matrix, pad_idx, pool, pretrained=pretrained).to(device)
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        best_loss, best_epoch, remaining = float('inf'), 0, args.patience
        best_path = RESULTS_DIR / f'{name}_seed{args.seed}.pt'
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device)
            all_history.append({'configuration': name, 'seed': args.seed, 'epoch': epoch,
                                **{f'train_{k}': v for k, v in train_metrics.items()},
                                **{f'val_{k}': v for k, v in val_metrics.items()}})
            print(f'{name:18s} epoch={epoch:02} '
                  f'val_loss={val_metrics["loss"]:.4f} '
                  f'val_micro_f1={val_metrics["micro_f1"]:.4f}')
            if val_metrics['loss'] < best_loss - 1e-5:
                best_loss, best_epoch, remaining = val_metrics['loss'], epoch, args.patience
                torch.save({'model_state': model.state_dict(), 'configuration': name,
                            'seed': args.seed, 'epoch': epoch}, best_path)
            else:
                remaining -= 1
                if remaining <= 0:
                    break
        # Re-evaluate only the selected checkpoint to avoid reporting the last epoch.
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:  # torch versions before weights_only support
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        selected = run_epoch(model, val_loader, device)
        summary.append({'configuration': name, 'pooling': pool, 'head': head,
                        'pretrained': pretrained, 'seed': args.seed,
                        'best_epoch': best_epoch,
                        **{f'val_{k}': v for k, v in selected.items()}})
        del model, optimizer
        if device == 'cuda':
            torch.cuda.empty_cache()

    pd.DataFrame(all_history).to_csv(RESULTS_DIR / f'history_seed{args.seed}.csv', index=False)
    pd.DataFrame(summary).to_csv(RESULTS_DIR / f'summary_seed{args.seed}.csv', index=False)
    with (RESULTS_DIR / f'settings_seed{args.seed}.json').open('w') as f:
        json.dump({**vars(args), 'device_used': device, 'compact_vocab_size': len(compact_vocab),
                   'validation_examples': len(val_ds), 'training_examples': len(train_ds)}, f, indent=2)
    print('\nValidation summary (not test results):')
    print(pd.DataFrame(summary).sort_values('val_micro_f1', ascending=False).to_string(index=False))
    print(f'\nSaved to: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
