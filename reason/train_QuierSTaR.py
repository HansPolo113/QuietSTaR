import os
import json
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MistralForCausalLM, LlamaTokenizerFast, MistralConfig
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from model import MistralForCausalLMWithQuietSTaR, MistralForQuietSTaR
from configuration import MistralConfigWithQuietSTaR, MistralConfigForQuietSTaR
from modeling_mistral_QuierSTaR import MistralForCausalLMWithQuietSTaROfficial


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
    n_ahead = 8
    n_ahead_talk = 4
    n_passes = 2
    gumbel_temperature = 1
    use_start_thought_token = True
    use_end_thought_token = True
    include_policy_loss = True
    gumbel_detach = True
    merged_talk_heads = True
    residual_think_head = False
    optimize_lm_head_only_at_start = False
    gradient_accumulation_steps = 1

    # config = MistralConfigWithQuietSTaR.from_pretrained("/home/emzhang/data/reason/nano-mistral",
    #                                                     use_cache=True,
    #                                                     max_thoughts=n_ahead + n_ahead_talk + 1,
    #                                                     merged_talk_heads=merged_talk_heads,
    #                                                     merged_lm_and_talk_heads=False,
    #                                                     merged_lm_and_think_heads=True, use_concat_talk_head=True,
    #                                                     use_shallow_think=True, use_shallow_talk=False,
    #                                                     use_complex_think_head=False, use_complex_talk_head=True,
    #                                                     use_weighted_talk_head=True, )

    config = MistralConfigForQuietSTaR.from_pretrained("/home/emzhang/data/reason/nano-mistral", n_true=4, n_thought=1,
                                                       thought_length=4)

    # model = MistralForCausalLMWithQuietSTaR.from_pretrained("/home/emzhang/data/reason/nano-mistral",
    #                                                         max_thoughts=n_ahead + n_ahead_talk + 1,
    #                                                         merged_talk_heads=merged_talk_heads,
    #                                                         merged_lm_and_talk_heads=False,
    #                                                         merged_lm_and_think_heads=True, use_concat_talk_head=True,
    #                                                         use_shallow_think=True, use_shallow_talk=False,
    #                                                         use_complex_think_head=False, use_complex_talk_head=True,
    #                                                         use_weighted_talk_head=True, )
    # model = MistralForCausalLMWithQuietSTaR.from_pretrained("/home/emzhang/data/reason/nano-mistral", config=config)
    model = MistralForQuietSTaR.from_pretrained("/home/emzhang/data/reason/nano-mistral", config=config)
    tokenizer = AutoTokenizer.from_pretrained("/home/emzhang/data/reason/nano-mistral")
    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens_to_add = []
    special_tokens_to_add.append("<|startthought|>")
    special_tokens_to_add.append("<|endthought|>")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))

    model.init_special_embedding(start_prompt="<|startthought|>", end_prompt="<|endthought|>", init_prompt="---",
                                 tokenizer=tokenizer)
    # model.tokenizer = tokenizer
    # model.gumbel_detach = gumbel_detach
    # model.include_policy_loss = include_policy_loss
    # model.use_end_thought_token = use_end_thought_token
    # model.use_start_thought_token = use_start_thought_token
    # model.n_ahead = n_ahead
    # model.n_ahead_talk = n_ahead_talk
    # model.n_passes = n_passes
    # model.n_tokens_print = gradient_accumulation_steps
    # model.gradient_accumulation_steps = gradient_accumulation_steps
    # model.residual_think_head = residual_think_head
    # model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    # model.gumbel_temperature = gumbel_temperature

    setting = ['standard', 'padding', 'reverse', 'paddingreverse'][1]

    if setting == 'standard':
        train_dataset = MultiplicationDataset(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer)
        val_dataset = MultiplicationDataset(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_QuietSTaR'

    elif setting == 'reverse':
        train_dataset = MultiplicationDatasetReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer)
        val_dataset = MultiplicationDatasetReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_reverse_QuietSTaR'

    elif setting == 'padding':
        train_dataset = MultiplicationDatasetPadding(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer, 8)
        val_dataset = MultiplicationDatasetPadding(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer, 8)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_padding_QuietSTaR'

    else:
        train_dataset = MultiplicationDatasetPaddingReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_train.json', tokenizer, 8)
        val_dataset = MultiplicationDatasetPaddingReverse(
            '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json', tokenizer, 8)
        output_dir = '/home/emzhang/code/nanoGPT/output/multiplication_8digit_paddingreverse_QuietSTaR'

    wandb.init(project='reason')

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=['none'],
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        bf16=True,
        save_total_limit=2,
        dataloader_num_workers=4,
        local_rank=int(os.getenv('LOCAL_RANK', -1))
    )
    # training_args.hub_project = ''
    model = model.to(dtype=torch.bfloat16)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()


if __name__ == '__main__':
    train()
