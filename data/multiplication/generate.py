import json
import random
from tqdm import tqdm

def generate_multiplication_data(max_digits, num_samples):
    data = []
    generated_samples = set()

    with tqdm(total=num_samples, unit='sample') as pbar:
        while len(data) < num_samples:
            # Randomly choose the number of digits for the multiplier and multiplicand
            multiplier_digits = random.randint(1, max_digits)
            multiplicand_digits = random.randint(1, max_digits)

            # Generate random multiplier and multiplicand
            multiplier = random.randint(10 ** (multiplier_digits - 1), 10 ** multiplier_digits - 1)
            multiplicand = random.randint(10 ** (multiplicand_digits - 1), 10 ** multiplicand_digits - 1)

            sample = (multiplier, multiplicand)
            if sample not in generated_samples:
                # Calculate the result
                result = multiplier * multiplicand

                # Append the sample to the data list
                data.append({
                    "multiplier": multiplier,
                    "multiplicand": multiplicand,
                    "result": result
                })
                generated_samples.add(sample)

                pbar.update(1)

    return data


# Generate the multiplication dataset
max_digits = 8
num_samples = 500000
dataset = generate_multiplication_data(max_digits, num_samples)

# Shuffle the dataset
random.shuffle(dataset)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
train_data = dataset[:train_size]
val_data = dataset[train_size:]

# Save the datasets as JSON files
with open(f"./multiplication_{max_digits}digit_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(f"multiplication_{max_digits}digit_val.json", "w") as f:
    json.dump(val_data, f, indent=2)

print("Dataset generated and saved successfully!")