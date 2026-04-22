"""
BERT fine-tuning module for SNLI.

Fine-tunes a pretrained BERT model on the Stanford Natural Language
Inference (SNLI) dataset, following the d2l methodology.
"""

import json
import multiprocessing
import os
import time

import torch
from torch import nn
from d2l import torch as d2l


# Register the pretrained BERT model in d2l's data hub
d2l.DATA_HUB["bert.small"] = (
    d2l.DATA_URL + "bert.small.torch.zip",
    "c72329e68a732bef0452e4b96a1c341c8910f81f",
)


def load_pretrained_bert(
    model_name="bert.small",
    hidden_units=256,
    feed_forward_units=512,
    attention_heads=4,
    num_blocks=2,
    dropout=0.1,
    max_sequence_length=512,
):
    """Load a pretrained BERT model and its vocabulary.

    Returns:
        tuple: (bert_model, vocabulary)
    """
    directory = d2l.download_extract(model_name)

    # Load vocabulary
    vocabulary = d2l.Vocab()
    vocab_path = os.path.join(directory, "vocab.json")
    vocabulary.idx_to_token = json.load(open(vocab_path))
    vocabulary.token_to_idx = {
        token: idx for idx, token in enumerate(vocabulary.idx_to_token)
    }

    # Initialize and load model weights
    bert_model = d2l.BERTModel(
        vocab_size=len(vocabulary),
        num_hiddens=hidden_units,
        ffn_num_hiddens=feed_forward_units,
        num_heads=attention_heads,
        num_blks=num_blocks,
        dropout=dropout,
        max_len=max_sequence_length,
    )
    params_path = os.path.join(directory, "pretrained.params")
    bert_model.load_state_dict(torch.load(params_path))

    return bert_model, vocabulary


def get_devices():
    """Get available computation devices (CUDA, MPS, or CPU fallback).

    Returns:
        list: List of torch.device objects.
    """
    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        print(f"Using CUDA GPU(s): {devices}")
        return devices
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS).")
        return [torch.device("mps")]
    print("No GPU found, using CPU.")
    return [torch.device("cpu")]


class SNLIBERTDataset(torch.utils.data.Dataset):
    """Dataset for SNLI with BERT tokenization.

    Tokenizes premise-hypothesis pairs, pads/truncates to a fixed
    sequence length, and stores token IDs, segment IDs, valid lengths,
    and labels.
    """

    def __init__(self, raw_dataset, max_sequence_length, vocabulary):
        # Tokenize and pair premises with hypotheses
        tokenized_pairs = [
            [premise, hypothesis]
            for premise, hypothesis in zip(
                *[
                    d2l.tokenize([s.lower() for s in sentences])
                    for sentences in raw_dataset[:2]
                ]
            )
        ]

        self.labels = torch.tensor(raw_dataset[2])
        self.vocab = vocabulary
        self.max_sequence_length = max_sequence_length

        self.all_token_ids, self.all_segment_ids, self.valid_lengths = (
            self._process_all_pairs(tokenized_pairs)
        )
        print(f"Read {len(self.all_token_ids)} examples")

    def _process_all_pairs(self, token_pairs):
        """Process all token pairs in parallel using multiprocessing."""
        with multiprocessing.Pool(4) as pool:
            results = pool.map(self._process_single_pair, token_pairs)

        token_ids = [r[0] for r in results]
        segment_ids = [r[1] for r in results]
        valid_lengths = [r[2] for r in results]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(valid_lengths),
        )

    def _process_single_pair(self, token_pair):
        """Process a single premise-hypothesis pair into BERT inputs."""
        premise_tokens, hypothesis_tokens = token_pair
        self._trim_to_max_length(premise_tokens, hypothesis_tokens)

        combined_tokens, segment_ids = d2l.get_tokens_and_segments(
            premise_tokens, hypothesis_tokens
        )

        # Pad to max sequence length
        pad_length = self.max_sequence_length - len(combined_tokens)
        padded_ids = self.vocab[combined_tokens] + [self.vocab["<pad>"]] * pad_length
        padded_segments = segment_ids + [0] * pad_length
        valid_length = len(combined_tokens)

        return padded_ids, padded_segments, valid_length

    def _trim_to_max_length(self, premise_tokens, hypothesis_tokens):
        """Trim the longer sequence until the pair fits within the limit.

        The limit accounts for 3 special tokens: [CLS], [SEP], [SEP].
        """
        max_tokens = self.max_sequence_length - 3
        while len(premise_tokens) + len(hypothesis_tokens) > max_tokens:
            if len(premise_tokens) > len(hypothesis_tokens):
                premise_tokens.pop()
            else:
                hypothesis_tokens.pop()

    def __getitem__(self, index):
        inputs = (
            self.all_token_ids[index],
            self.all_segment_ids[index],
            self.valid_lengths[index],
        )
        return inputs, self.labels[index]

    def __len__(self):
        return len(self.all_token_ids)


class BERTClassifier(nn.Module):
    """3-class classifier built on top of a pretrained BERT encoder."""

    def __init__(self, pretrained_bert):
        super().__init__()
        self.encoder = pretrained_bert.encoder
        self.hidden = pretrained_bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        token_ids, segment_ids, valid_lengths = inputs
        encoded = self.encoder(token_ids, segment_ids, valid_lengths)
        # Use the [CLS] token representation for classification
        return self.output(self.hidden(encoded[:, 0, :]))


def prepare_snli_data(vocabulary, batch_size=256, max_sequence_length=128):
    """Download SNLI and create DataLoaders for training and testing.

    Returns:
        tuple: (train_loader, test_loader)
    """
    worker_count = d2l.get_dataloader_workers()
    dataset_dir = d2l.download_extract("SNLI")

    train_dataset = SNLIBERTDataset(
        d2l.read_snli(dataset_dir, True), max_sequence_length, vocabulary
    )
    test_dataset = SNLIBERTDataset(
        d2l.read_snli(dataset_dir, False), max_sequence_length, vocabulary
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=worker_count
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size, num_workers=worker_count
    )

    return train_loader, test_loader


def _evaluate_accuracy(net, data_loader, device):
    """Compute classification accuracy of ``net`` over ``data_loader``."""
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in data_loader:
            if isinstance(features, list):
                features = [x.to(device) for x in features]
            else:
                features = features.to(device)
            labels = labels.to(device)
            preds = net(features)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return correct / total


def train_classifier(bert_model, devices, learning_rate=1e-4, epochs=6):
    """Fine-tune BERT on SNLI and report training time.

    Returns:
        BERTClassifier: The trained classifier model.
    """
    _, vocabulary = load_pretrained_bert()
    train_loader, test_loader = prepare_snli_data(vocabulary)

    classifier = BERTClassifier(bert_model)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Warm up the model with a single batch (runs on CPU to initialise LazyLinear)
    classifier(next(iter(train_loader))[0])
    device = devices[0]
    if device.type == "cuda" and len(devices) > 1:
        net = nn.DataParallel(classifier, device_ids=list(range(len(devices)))).to(device)
    else:
        net = classifier.to(device)

    num_batches = len(train_loader)
    total_batches = num_batches * epochs
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss, running_correct, running_seen = 0.0, 0, 0
        net.train()

        for i, (features, labels) in enumerate(train_loader):
            if isinstance(features, list):
                features = [x.to(device) for x in features]
            else:
                features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = net(features)
            loss = loss_fn(preds, labels)
            loss.sum().backward()
            optimizer.step()

            running_loss += loss.sum().item()
            running_correct += (preds.argmax(dim=1) == labels).sum().item()
            running_seen += labels.numel()

            done = epoch * num_batches + i + 1
            elapsed = time.time() - start_time
            eta = elapsed * (total_batches - done) / done
            print(
                f"\rEpoch {epoch + 1}/{epochs} "
                f"batch {i + 1}/{num_batches} "
                f"loss={running_loss / running_seen:.4f} "
                f"acc={running_correct / running_seen:.4f} "
                f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                end="",
                flush=True,
            )

        test_acc = _evaluate_accuracy(net, test_loader, device)
        print(
            f"\nEpoch {epoch + 1} done — "
            f"train_loss={running_loss / running_seen:.4f} "
            f"train_acc={running_correct / running_seen:.4f} "
            f"test_acc={test_acc:.4f} "
            f"({(time.time() - epoch_start) / 60:.2f}m)"
        )

    elapsed_minutes = (time.time() - start_time) / 60

    print(f"Training took: {elapsed_minutes:.2f} minutes")
    print("Accuracy is ~0.8 on test sets.")
    print("A higher-accuracy NLI model will be used for the analysis.")

    return classifier
