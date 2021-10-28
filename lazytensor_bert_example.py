"""
Runs a training of the Bert model using the Lazy Tensor Core with the
TorchScript backend.

Requirements to run example:
- python -m pip install transformers datasets
- `lazy_tensor_core` Python package
    For information on how to obtain the `lazy_tensor_core` Python package,
    see here:

    https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/QUICKSTART.md

To run the example, make sure `/path/to/pytorch/lazy_tensor_core` is in your
PYTHONPATH. Then, run

    python lazytensor_bert_example.py

The output of this example can be found in
    `lazytensor_bert_example_output.txt`

Most of the code in this example was copied from the wonderful tutorial
    https://huggingface.co/transformers/training.html#fine-tuning-in-native-pytorch
"""
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, \
    BertTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics


def tokenize_dataset(dataset: DatasetDict) -> DatasetDict:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    return tokenized_datasets


def train(model: BertForSequenceClassification,
          num_epochs: int,
          num_training_steps: int,
          train_dataloader: DataLoader,
          device: torch.device) -> List[torch.Tensor]:
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    model.train()
    losses = []
    for _ in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            losses.append(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    return losses


def main():
    ltc._LAZYC._ltc_init_ts_backend()
    device = torch.device('lazy')

    tokenized_datasets = tokenize_dataset(load_dataset('imdb'))
    small_train_dataset = tokenized_datasets['train'].shuffle(seed=42)\
                                                     .select(range(2))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True,
                                  batch_size=8)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                          num_labels=2)
    model.to(device)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    losses = train(model, num_epochs,
                   num_training_steps, train_dataloader, device)

    print('\nMetrics report:')
    print(metrics.metrics_report())

    print('\nTorchScriptGraph:')
    graph_str = ltc._LAZYC._get_ltc_tensors_backend([losses[0]])
    print(graph_str)


if __name__ == '__main__':
    main()
