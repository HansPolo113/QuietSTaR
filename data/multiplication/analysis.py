import json
import numpy as np


def analyze_multiplication_data(file_path, max_digits):
    with open(file_path, 'r') as f:
        data = json.load(f)

    output = np.zeros((max_digits, max_digits), dtype=int)

    for sample in data:
        multiplier = sample['multiplier']
        multiplicand = sample['multiplicand']

        multiplier_digits = len(str(multiplier))
        multiplicand_digits = len(str(multiplicand))

        output[multiplier_digits - 1][multiplicand_digits - 1] += 1

    return output


# Analyze the training dataset
train_file_path = './multiplication_8digit_train.json'
max_digits = 8
train_output = analyze_multiplication_data(train_file_path, max_digits)

print("Training Dataset Analysis:")
print(train_output)
print("Total samples in the training dataset:", np.sum(train_output))

# Analyze the validation dataset
val_file_path = './multiplication_8digit_val.json'
val_output = analyze_multiplication_data(val_file_path, max_digits)

print("\nValidation Dataset Analysis:")
print(val_output)
print("Total samples in the validation dataset:", np.sum(val_output))