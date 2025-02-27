"""

SBERTMetric class:
-------------------------------------------------------
Class for calculating SentenceBERT similarity on AttackResults

"""

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.constraints.semantics.sentence_encoders import BERT
from textattack.metrics import Metric


class SBERTMetric(Metric):
    def __init__(
            self,
            model_name="stsb-mpnet-base-v2",    # best model from https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
            **kwargs
            ):
        self.model = BERT(model_name=model_name,  metric="cosine")
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates average Sentence BERT similarity on all successfull attacks.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset

        Example::


            >> import textattack
            >> import transformers
            >> model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
            >> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
            >> dataset = textattack.datasets.HuggingFaceDataset("glue", "sst2", split="train")
            >> attack_args = textattack.AttackArgs(
                num_examples=1,
                log_to_csv="log.csv",
                checkpoint_interval=5,
                checkpoint_dir="checkpoints",
                disable_stdout=True
            )
            >> attacker = textattack.Attacker(attack, dataset, attack_args)
            >> results = attacker.attack_dataset()
            >> sbertm = textattack.metrics.quality_metrics.SBERTMetric().calculate(results)
        """

        self.results = results

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text)
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text)

        sbert_scores = []
        for c in range(len(self.original_candidates)):
            sbert_scores.append(
                self.model._sim_score(
                    self.original_candidates[c], self.successful_candidates[c]
                ).item()
            )

        self.all_metrics["avg_sentence_bert_similarity"] = round(
            sum(sbert_scores) / len(sbert_scores), 3
        ) if len(sbert_scores) > 0 else 0.0

        return self.all_metrics

