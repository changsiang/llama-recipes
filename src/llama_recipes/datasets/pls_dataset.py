import csv
from llama_recipes.datasets.utils import Concatenator

def get_preprocessed_pls(dataset_config, tokenizer, split):
    dataset = []
    with open('./dataset/fullset.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            dataset.append(row)

    prompt = (
        f"Summarize this medical text:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["input_text"],
                summary=sample["target_text"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset