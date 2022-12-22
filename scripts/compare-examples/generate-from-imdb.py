# use local implementation of textattack
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import transformers

from textattack.attack_recipes import SynBA2022, TextFoolerJin2019, BAEGarg2019

from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

from textattack import Attacker
from textattack import AttackArgs

# Import the model
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# import the dataset
dataset_name = "imdb"
dataset = HuggingFaceDataset(dataset_name, None, "test")

# choose the attack method
attack_class = BAEGarg2019
attack = attack_class.build(model_wrapper)

# whether to make the attack reproducible or set it random for each run
random_seed = 765

# set the number of samples to attack
num_examples = 100

# set the log file name where the results will be saved
log_file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs", dataset_name, f"log_{attack_class.__name__}_{dataset_name}.csv")
print(f"Log file: {log_file_name}")

# start the attack
attack_args = AttackArgs(num_examples=num_examples, shuffle=True, random_seed=random_seed, csv_coloring_style="html", disable_stdout=False, log_to_csv=log_file_name)
attacker = Attacker(attack, dataset, attack_args)
attack_results = attacker.attack_dataset()
