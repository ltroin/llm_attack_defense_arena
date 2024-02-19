import random
import string
from typing import Type

from datasets import load_dataset, get_dataset_config_names


class Benchmark:
    """Base class for different huggingface datasets benchmarks to use.  Handles the formatting of questions and evaluating the correctness of answers"""

    repo = ""
    test_split = ""
    held_out_split = ""

    def __init__(self, config_name: str, split: list[str] | None = None):

        """
        Args:
            config_name: The name of the benchmark config ('all', 'algebra', 'sports', etc.)
            split: The name of the split to use ('train', 'dev', 'test', etc.)"""

        if split is None:
            split = [self.test_split]
        if self.held_out_split not in split:
            split.append(self.held_out_split)

        split_dataset = load_dataset(self.repo, config_name, split=split)
        self.dataset = {split_name: split_dataset[split_idx] for split_idx, split_name in enumerate(split)}
        self.split = split

    @classmethod
    def configs(cls):
        """Gets the valid configs for this benchmark

        Returns:
            The valid configs"""

        return get_dataset_config_names(cls.repo)

    def split_prompts(self, split_name: str = None):
        """Selects the requested split of the dataset. The default split is used if none is provided

        Args:
            split_name: The name of the split to select
        Returns:
            The split of the dataset"""

        if split_name is None:
            split_name = self.split[0]

        return self.dataset[split_name]

    def format_question(self, question_id: int, split_name: str = "", n_shot=0, as_example=False) -> str:
        """Formats a question with the given id as an n-shot prompt to be given directly to a model for answering

        Args:
            question_id: The id of the question to format
            split_name: The name of the split that the question is from
            n_shot: The number of examples to give before the actual question
            as_example: Whether to generate only the question itself as an example if true or to add prolog and epilog prompts if false

        Returns:
            The formatted question"""

        ...

    def batch_format_questions(self, split_name: str = "", n_shot=0):
        """Formats all questions within a given split of the dataset

        Args:
            split_name: The name of the split to select
            n_shot: The number of examples given per question

        Returns:
            A list of all formatted questions"""

        num_prompts = len(self.split_prompts(split_name))
        return [self.format_question(i, split_name=split_name, n_shot=n_shot, as_example=False) for i in range(num_prompts)]

    def evaluate_answer(self, question_id: int, model_response: str, split_name: str = "") -> bool:
        """Evaluates whether an answer to the given question is correct

        Args:
            question_id: The id of the question to compare against
            model_response: The answer response from the model
            split_name: The split of the dataset that the question is from
        Returns:
            Whether the given answer can be considered correct"""

        ...

    def batch_evaluate_answers(self, model_responses: list[str], split_name: str = ""):
        """Evaluates all questions within the given dataset split

        Args:
            model_responses: The model's responses to all questions within the split
            split_name: The name of the split that the questions are from
        Returns:
            A list of boolean correctness evaluations"""

        return [self.evaluate_answer(i, model_responses[i], split_name=split_name) for i in range(len(model_responses))]


class MMLU(Benchmark):

    repo = "cais/mmlu"
    test_split = "test"
    held_out_split = "validation"

    @classmethod
    def configs(cls):
        return [config for config in super().configs() if config != "all"]

    def format_question(self, question_id: int, split_name: str = "", n_shot=0, as_example=False) -> str:

        # Collect few-shot examples
        examples = []
        num_dev = len(self.dataset[self.held_out_split])
        while len(examples) < n_shot:
            examples.append(self.format_question(random.randint(0, num_dev-1), split_name=self.held_out_split, n_shot=0, as_example=True))

        # Format question as the question itself, newline-seperated choices, and a prompt for an answer
        sample = self.split_prompts(split_name=split_name)[question_id]
        choices = "\n".join([f"{string.ascii_lowercase[i]}) {choice}" for i, choice in enumerate(sample["choices"])])
        question = f"""{sample["question"]}
{choices}
Answer: """

        # Include the answer to the prompt only if it is an example
        if as_example:
            question += string.ascii_lowercase[sample["answer"]]

        # Create question-answering prompt
        prompt = "" if as_example else "Please answer the following question to the best of your ability. In your output give the answer ONLY. Do not give any explanation or other text. Just the letter choice if your answer."
        if len(examples) > 0:
            prompt += "\nFor example:\n"+"\n\n".join(examples)
        prompt += ("\n\nNow, answer this question ONLY, nothing else.\n" if not as_example else "")+question

        return prompt

    def evaluate_answer(self, question_id: int, model_response: str, split_name: str = ""):

        answer = string.ascii_lowercase[self.split_prompts(split_name=split_name)[question_id]["answer"]]
        model_response = model_response.replace(")", " ").strip()

        return model_response[0] == answer


def benchmark_from_name(benchmark_name: str, config_name: str = None, split: list[str] | None = None) -> Benchmark:
    """Gets an instance of a benchmark from its name

    Args:
        benchmark_name: The name of the benchmark
        config_name: The configuration for that benchmark
        split: The split, if any, to get
    Returns:
        An instance of the requested benchmark"""

    if "mmlu" in benchmark_name.lower():
        return benchmark_class_from_name(benchmark_name)(config_name=config_name, split=split)

    raise ValueError("Could not find benchmark "+benchmark_name)


def benchmark_class_from_name(benchmark_name: str) -> Type[Benchmark]:
    """Gets the class of a benchmark from its name

    Args:
        benchmark_name: The name of the benchmark
    Returns:
        The class of the requested benchmark"""

    if "mmlu" in benchmark_name.lower():
        return MMLU

    raise ValueError("Could not find benchmark "+benchmark_name)

