import os
import json
import numpy as np
import pandas as pd
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerState, TrainerControl
from transformers import MistralForCausalLM, LlamaTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback

from tqdm import tqdm


class MultiplicationAccuracyCallback(TrainerCallback):
    def __init__(self, num_digits=8):
        super().__init__()
        self.correct_array = np.zeros((num_digits, num_digits), dtype=int)
        self.total_array = np.zeros((num_digits, num_digits), dtype=int)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logits=None,
                           labels=None, **kwargs):
        predictions = logits.argmax(-1) if logits is not None else None
        inputs = kwargs.get('inputs')

        for i in range(inputs['input_ids'].size(0)):
            predicted_text = tokenizer.decode(predictions[i], skip_special_tokens=True).strip()
            true_result = str(inputs['result'][i])

            is_correct = predicted_text == true_result

            multiplier_digits = len(str(inputs['multiplier'][i]))
            multiplicand_digits = len(str(inputs['multiplicand'][i]))

            self.correct_array[multiplier_digits - 1, multiplicand_digits - 1] += int(is_correct)
            self.total_array[multiplier_digits - 1, multiplicand_digits - 1] += 1

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        accuracy_matrix = np.divide(self.correct_array, self.total_array, where=self.total_array != 0)
        print("Accuracy Matrix:", accuracy_matrix)


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
        # target_text = f"{' '.join(str(result))}"
        # total_input_text = input_text + ' ' + target_text

        input_encodings = self.tokenizer(input_text, truncation=True, max_length=128, return_tensors='pt')
        # target_encodings = self.tokenizer(target_text, truncation=True, max_length=128, return_tensors='pt')
        # total_encodings = self.tokenizer(total_input_text, truncation=True, max_length=128, return_tensors='pt')

        input_ids = input_encodings['input_ids'].squeeze(0)
        attention_mask = input_encodings['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'multiplier': multiplier,
            'multiplicand': multiplicand,
            'result': result,
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
        # target_text = f"{' '.join(str(result))}"
        # total_input_text = input_text + ' ' + target_text

        input_encodings = self.tokenizer(input_text, truncation=True, max_length=128, return_tensors='pt')
        # target_encodings = self.tokenizer(target_text, truncation=True, max_length=128, return_tensors='pt')
        # total_encodings = self.tokenizer(total_input_text, truncation=True, max_length=128, return_tensors='pt')

        input_ids = input_encodings['input_ids'].squeeze(0)
        attention_mask = input_encodings['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'multiplier': multiplier,
            'multiplicand': multiplicand,
            'result': result,
        }

tokenizer = AutoTokenizer.from_pretrained("/home/emzhang/data/reason/nano-mistral")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def collate_fn(batch):
    def left_pad(tensor_list, padding_value):
        reversed_tensors = [torch.flip(t, [0]) for t in tensor_list]
        padded = pad_sequence(reversed_tensors, batch_first=True, padding_value=padding_value)
        return torch.flip(padded, [1])

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    padded_input_ids = left_pad(input_ids, tokenizer.pad_token_id)
    padded_attention_masks = left_pad(attention_masks, 0)

    multipliers = [item['multiplier'] for item in batch]
    multiplicands = [item['multiplicand'] for item in batch]
    results = [item['result'] for item in batch]

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'multiplier': multipliers,
        'multiplicand': multiplicands,
        'result': results
    }


def generate_answer(model, tokenizer, input_ids, attention_mask):
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    outputs = model.generate(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0),
                             max_new_tokens=64)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_part = tokenizer.decode(input_ids, skip_special_tokens=True)
    answer_start_index = len(input_part) + 1
    return full_text[answer_start_index:]


def evaluate_model(model, dataloader, num_digits, device, setting):
    model.eval()
    correct_matrix = np.zeros((num_digits, num_digits), dtype=int)
    total_matrix = np.zeros((num_digits, num_digits), dtype=int)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
            input_length = input_ids.shape[-1]
            outputs = outputs[:, input_length:]
            results = [tokenizer.decode(output, skip_special_tokens=True).strip().replace(' ', '') for output in outputs]
            if setting == 'padding':
                results = [str(int(result)) for result in results]
            if setting == 'reverse':
                results = [result[::-1] for result in results]
            if setting == 'paddingreverse':
                results = [result[::-1] for result in results]
                results = [str(int(result)) for result in results]
            expected_results = [str(result).strip() for result in batch['result']]

            for idx, (result, expected) in enumerate(zip(results, expected_results)):
                is_correct = result == expected
                multiplier = str(batch['multiplier'][idx])
                multiplicand = str(batch['multiplicand'][idx])

                multiplier_len = len(multiplier) - 1
                multiplicand_len = len(multiplicand) - 1

                correct_matrix[multiplier_len, multiplicand_len] += int(is_correct)
                total_matrix[multiplier_len, multiplicand_len] += 1
    accuracy_matrix = np.divide(correct_matrix, total_matrix, where=total_matrix != 0)
    return accuracy_matrix


def evaluation():
    num_digit = 8
    setting = ['standard', 'padding', 'reverse', 'paddingreverse'][3]
    if setting == 'standard':
        model_path = '/home/emzhang/code/nanoGPT/output/multiplication_8digit/checkpoint-3910'
    else:
        model_path = f'/home/emzhang/code/nanoGPT/output/multiplication_8digit_{setting}/checkpoint-3910'

    model = MistralForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("/home/emzhang/data/reason/nano-mistral")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    eval_path = '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json'
    if setting == 'standard' or setting == 'reverse':
        eval_dataset = MultiplicationDataset(eval_path, tokenizer)
    else:
        eval_dataset = MultiplicationDatasetPadding(eval_path, tokenizer, num_digit)

    eval_loader = DataLoader(eval_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')

    accuracy_matrix = evaluate_model(model, eval_loader, num_digit, device, setting)
    print(accuracy_matrix)
    df = pd.DataFrame(accuracy_matrix)
    if setting == 'standard':
        save_path = '/home/emzhang/code/nanoGPT/output/multiplication_8digit/accuracy_matrix.csv'
    else:
        save_path = f'/home/emzhang/code/nanoGPT/output/multiplication_8digit_{setting}/accuracy_matrix.csv'
    df = df.round(5)
    df.to_csv(save_path, index=False)

    # training_args = TrainingArguments(
    #     output_dir='/home/emzhang/code/nanoGPT/output/multiplication_8digit',
    #     per_device_eval_batch_size=16,
    #     do_predict=True,
    #     report_to=['none'],
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     data_collator=collate_fn,
    #     callbacks=[MultiplicationAccuracyCallback(num_digits=num_digit)],
    # )

    # trainer.predict(eval_dataset)

    # model.eval()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #
    # eval_path = '/home/emzhang/code/nanoGPT/data/multiplication/multiplication_8digit_val.json'
    # eval_dataset = MultiplicationDataset(eval_path, tokenizer)
    #
    # correct_array = np.zeros((num_digit, num_digit), dtype=int)
    # total_array = np.zeros((num_digit, num_digit), dtype=int)
    #
    # for item in tqdm(eval_dataset):
    #     generated_result = generate_answer(model, tokenizer, item['input_ids'], item['attention_mask'])
    #
    #     is_correct = generated_result.strip() == str(item['result'])
    #     multiplier_digits = len(str(item['multiplier']))
    #     multiplicand_digits = len(str(item['multiplicand']))
    #
    #     correct_array[multiplier_digits - 1, multiplicand_digits - 1] += int(is_correct)
    #     total_array[multiplier_digits - 1, multiplicand_digits - 1] += 1
    #
    # accuracy_matrix = np.divide(correct_array, total_array, where=total_array != 0)
    # return accuracy_matrix


if __name__ == '__main__':
    evaluation()
