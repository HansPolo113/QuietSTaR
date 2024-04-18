import os
import json
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MistralForCausalLM, LlamaTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from model import MistralForCausalLMWithQuietSTaR


class MultiplicationDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        multiplier = sample['multiplier']
        multiplicand = sample['multiplicand']
        result = sample['result']

        input_text = f"{' '.join(str(multiplier))} * {' '.join(str(multiplicand))} ="
        target_text = f"{' '.join(str(result))}"
        total_input_text = input_text + ' ' + target_text

        input_encodings = self.tokenizer(input_text, truncation=True, max_length=128, return_tensors='pt')
        target_encodings = self.tokenizer(target_text, truncation=True, max_length=128, return_tensors='pt',
                                          add_special_tokens=False)
        total_encodings = self.tokenizer(total_input_text, truncation=True, max_length=128, return_tensors='pt')

        input_ids = total_encodings['input_ids'].squeeze(0)
        attention_mask = total_encodings['attention_mask'].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        # + 1 for eos token
        target_len = len(target_encodings['input_ids'][0]) + 1
        labels[-target_len:] = input_ids[-target_len:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class MultiplicationDatasetReverse(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        multiplier = sample['multiplier']
        multiplicand = sample['multiplicand']
        result = sample['result']

        input_text = f"{' '.join(str(multiplier))} * {' '.join(str(multiplicand))} ="
        target_text = f"{' '.join(str(result)[::-1])}"
        total_input_text = input_text + ' ' + target_text

        input_encodings = self.tokenizer(input_text, truncation=True, max_length=128, return_tensors='pt')
        target_encodings = self.tokenizer(target_text, truncation=True, max_length=128, return_tensors='pt',
                                          add_special_tokens=False)
        total_encodings = self.tokenizer(total_input_text, truncation=True, max_length=128, return_tensors='pt')

        input_ids = total_encodings['input_ids'].squeeze(0)
        attention_mask = total_encodings['attention_mask'].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        # + 1 for eos token
        target_len = len(target_encodings['input_ids'][0]) + 1
        labels[-target_len:] = input_ids[-target_len:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class MultiplicationDatasetPadding(Dataset):
    def __init__(self, file_path, tokenizer, num_digits):
        with open(file_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.num_digits = num_digits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        multiplier = sample['multiplier']
        multiplicand = sample['multiplicand']
        result = sample['result']

        input_text = f"{' '.join(str(multiplier).zfill(self.num_digits))} * {' '.join(str(multiplicand).zfill(self.num_digits))} ="
        target_text = f"{' '.join(str(result).zfill(2 * self.num_digits))}"
        total_input_text = input_text + ' ' + target_text

        input_encodings = self.tokenizer(input_text, truncation=True, max_length=128, return_tensors='pt')
        target_encodings = self.tokenizer(target_text, truncation=True, max_length=128, return_tensors='pt',
                                          add_special_tokens=False)
        total_encodings = self.tokenizer(total_input_text, truncation=True, max_length=128, return_tensors='pt')

        input_ids = total_encodings['input_ids'].squeeze(0)
        attention_mask = total_encodings['attention_mask'].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        # + 1 for eos token
        target_len = len(target_encodings['input_ids'][0]) + 1
        labels[-target_len:] = input_ids[-target_len:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class MultiplicationDatasetPaddingReverse(Dataset):
    def __init__(self, file_path, tokenizer, num_digits):
        with open(file_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.num_digits = num_digits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        multiplier = sample['multiplier']
        multiplicand = sample['multiplicand']
        result = sample['result']

        input_text = f"{' '.join(str(multiplier).zfill(self.num_digits))} * {' '.join(str(multiplicand).zfill(self.num_digits))} ="
        target_text = f"{' '.join(str(result).zfill(2 * self.num_digits)[::-1])}"
        total_input_text = input_text + ' ' + target_text

        input_encodings = self.tokenizer(input_text, truncation=True, max_length=128, return_tensors='pt')
        target_encodings = self.tokenizer(target_text, truncation=True, max_length=128, return_tensors='pt',
                                          add_special_tokens=False)
        total_encodings = self.tokenizer(total_input_text, truncation=True, max_length=128, return_tensors='pt')

        input_ids = total_encodings['input_ids'].squeeze(0)
        attention_mask = total_encodings['attention_mask'].squeeze(0)
        labels = torch.full_like(input_ids, fill_value=-100)
        # + 1 for eos token
        target_len = len(target_encodings['input_ids'][0]) + 1
        labels[-target_len:] = input_ids[-target_len:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


tokenizer = AutoTokenizer.from_pretrained("/home/emzhang/data/reason/nano-mistral")


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded
    }


def train():
    n_ahead = 12
    n_ahead_talk = 4
    n_passes = 2
    gumbel_temperature = 1
    model = MistralForCausalLM.from_pretrained("/home/emzhang/data/reason/nano-mistral")
    tokenizer = AutoTokenizer.from_pretrained("/home/emzhang/data/reason/nano-mistral")
    tokenizer.add_eos_token = True

    setting = ['standard', 'padding', 'reverse', 'paddingreverse'][0]

    if setting == 'standard':
        train_dataset = MultiplicationDataset('/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer)
        val_dataset = MultiplicationDataset('/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit'

    if setting == 'reverse':
        train_dataset = MultiplicationDatasetReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer)
        val_dataset = MultiplicationDatasetReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_reverse'

    if setting == 'padding':
        train_dataset = MultiplicationDatasetPadding(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer, 8)
        val_dataset = MultiplicationDatasetPadding(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer, 8)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_padding'

    if setting == 'paddingreverse':
        train_dataset = MultiplicationDatasetPaddingReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer, 8)
        val_dataset = MultiplicationDatasetPaddingReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer, 8)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_paddingreverse'

    wandb.init(project='reason')

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=['wandb'],
        num_train_epochs=5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        dataloader_num_workers=4,
        local_rank=int(os.getenv('LOCAL_RANK', -1))
    )
    # training_args.hub_project = ''

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    trainer.train()


if __name__ == '__main__':
    train()
