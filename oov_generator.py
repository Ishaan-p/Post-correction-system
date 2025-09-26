from googletrans import Translator
from text_to_num import text2num
import re
from datasets import load_dataset
import os
import csv
import pandas as pd

dataset = iter(load_dataset("ai4bharat/IndicVoices","hindi",split="train", streaming=True))


def has_word_numbers(sentence):
    # Split sentence into words
    words = re.findall(r'\b\w+\b', sentence.lower())
    
    # Check all possible contiguous word sequences up to length 4
    # (to catch numbers like "twenty five" or "one hundred")
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words))+1):
            phrase = ' '.join(words[i:j])
            try:
                text2num(phrase, "en")
                return True  # If conversion succeeds, there is a number
            except ValueError:
                continue
    return False

def create_oov_audio_list():
    translator = Translator()

    verbatims = []
    audio_files = []
    sample_rates = []
    collected_files = 0

    while collected_files < 6:
        sample = next(dataset)
        translated = translator.translate(sample["verbatim"], src="hi", dest="en").text
        is_translated = has_word_numbers(translated)
        if is_translated == True:
           verbatims.append(sample["verbatim"])
           audio_files.append(sample["audio_filepath"]["array"])
           sample_rates.append(sample["audio_filepath"]["sampling_rate"])
           collected_files = collected_files + 1

    print(verbatims)
    print(sample_rates)
    return audio_files


def create_database_new(batch_data, csv_file="oov.csv"):
    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    # Append new rows
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header if file didn't exist
        if not file_exists:
            writer.writerow(["id", "audio"])

        # Figure out the next ID
        if file_exists:
            with open(csv_file, mode="r", encoding="utf-8") as fr:
                reader = list(csv.reader(fr))
                next_id = len(reader)  # header counts as row 0
        else:
            next_id = 1

        # Write each tuple in batch_data
        for i in range(0, len(batch_data)):
            print("Row:", batch_data[i])
            print("Listed Row:", batch_data[i])
            writer.writerow([next_id] + batch_data[i].tolist())
            next_id += 1

    # Read the CSV back into batch_data-style list
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = [(row["id"], row["audio"]) for row in reader]

    return results

data = create_oov_audio_list()
create_database_new(data)

def return_database():
    reference = pd.read_csv("./oov.csv")
    oov_data = []
    for _ in range(6):
        reference_row = reference[reference["id"] == _+1]
        oov_data.append(reference_row["audio"].values[0])

    return oov_data


print(return_database())
