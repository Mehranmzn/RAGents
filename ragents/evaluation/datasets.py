"""Evaluation dataset utilities and fixtures."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .types import EvaluationDataPoint


class EvaluationDataset:
    """Container for evaluation datasets."""

    def __init__(self, name: str, data_points: List[EvaluationDataPoint]):
        self.name = name
        self.data_points = data_points

    def __len__(self) -> int:
        return len(self.data_points)

    def __getitem__(self, index: int) -> EvaluationDataPoint:
        return self.data_points[index]

    def sample(self, n: int, random_state: Optional[int] = None) -> "EvaluationDataset":
        """Sample n data points from the dataset."""
        if random_state is not None:
            random.seed(random_state)

        sampled = random.sample(self.data_points, min(n, len(self.data_points)))
        return EvaluationDataset(f"{self.name}_sample_{n}", sampled)

    def split(self, test_size: float = 0.2, random_state: Optional[int] = None) -> tuple["EvaluationDataset", "EvaluationDataset"]:
        """Split dataset into train and test sets."""
        if random_state is not None:
            random.seed(random_state)

        shuffled = self.data_points.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_size))
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]

        return (
            EvaluationDataset(f"{self.name}_train", train_data),
            EvaluationDataset(f"{self.name}_test", test_data)
        )

    def filter_by_metadata(self, key: str, value: str) -> "EvaluationDataset":
        """Filter dataset by metadata values."""
        filtered = [
            dp for dp in self.data_points
            if dp.metadata.get(key) == value
        ]
        return EvaluationDataset(f"{self.name}_filtered_{key}_{value}", filtered)

    def save_to_json(self, file_path: str) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "data_points": [dp.model_dump() for dp in self.data_points]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        data_points = [EvaluationDataPoint(**dp) for dp in data["data_points"]]
        return cls(data["name"], data_points)

    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics."""
        stats = {
            "name": self.name,
            "size": len(self.data_points),
            "avg_question_length": 0,
            "avg_ground_truth_length": 0,
            "avg_contexts_count": 0,
            "metadata_keys": set(),
        }

        if self.data_points:
            stats["avg_question_length"] = sum(len(dp.question) for dp in self.data_points) / len(self.data_points)
            stats["avg_ground_truth_length"] = sum(len(dp.ground_truth) for dp in self.data_points) / len(self.data_points)
            stats["avg_contexts_count"] = sum(len(dp.contexts) for dp in self.data_points) / len(self.data_points)

            for dp in self.data_points:
                stats["metadata_keys"].update(dp.metadata.keys())

        stats["metadata_keys"] = list(stats["metadata_keys"])
        return stats


def create_sample_dataset(domain: str = "general") -> EvaluationDataset:
    """Create sample evaluation datasets for different domains."""

    if domain == "general":
        return _create_general_dataset()
    elif domain == "science":
        return _create_science_dataset()
    elif domain == "history":
        return _create_history_dataset()
    elif domain == "technology":
        return _create_technology_dataset()
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _create_general_dataset() -> EvaluationDataset:
    """Create a general knowledge evaluation dataset."""
    data_points = [
        EvaluationDataPoint(
            question="What is the capital of France?",
            ground_truth="The capital of France is Paris. Paris is located in the north-central part of France and is the country's largest city.",
            contexts=[
                "Paris is the capital and most populous city of France. Located in the north-central part of the country.",
                "France is a country in Western Europe with several cities, but Paris serves as its capital.",
            ],
            metadata={"domain": "geography", "difficulty": "easy"}
        ),
        EvaluationDataPoint(
            question="How does photosynthesis work?",
            ground_truth="Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It occurs in chloroplasts and involves two main stages: light-dependent reactions and light-independent reactions (Calvin cycle).",
            contexts=[
                "Photosynthesis is a process used by plants to convert light energy into chemical energy stored in glucose.",
                "The process occurs in chloroplasts and requires sunlight, carbon dioxide, and water as inputs.",
                "The products of photosynthesis are glucose (sugar) and oxygen gas.",
            ],
            metadata={"domain": "biology", "difficulty": "medium"}
        ),
        EvaluationDataPoint(
            question="What causes climate change?",
            ground_truth="Climate change is primarily caused by human activities that increase greenhouse gas concentrations in the atmosphere, particularly burning fossil fuels, deforestation, and industrial processes. These activities trap more heat in Earth's atmosphere, leading to global warming.",
            contexts=[
                "Greenhouse gases like carbon dioxide trap heat in the atmosphere, causing global warming.",
                "Burning fossil fuels releases carbon dioxide and other greenhouse gases into the atmosphere.",
                "Deforestation reduces the Earth's capacity to absorb carbon dioxide from the atmosphere.",
                "Industrial processes and agriculture also contribute to greenhouse gas emissions.",
            ],
            metadata={"domain": "environmental_science", "difficulty": "medium"}
        ),
        EvaluationDataPoint(
            question="Who wrote Romeo and Juliet?",
            ground_truth="Romeo and Juliet was written by William Shakespeare. It is one of his most famous tragedies, written in the early part of his career, around 1594-1596.",
            contexts=[
                "William Shakespeare was an English playwright and poet who lived from 1564 to 1616.",
                "Romeo and Juliet is a tragedy written by Shakespeare in the 1590s.",
                "The play tells the story of two young star-crossed lovers whose deaths unite their feuding families.",
            ],
            metadata={"domain": "literature", "difficulty": "easy"}
        ),
        EvaluationDataPoint(
            question="Explain the theory of relativity.",
            ground_truth="Einstein's theory of relativity consists of special relativity (1905) and general relativity (1915). Special relativity shows that space and time are interwoven and relative to the observer's motion, with the speed of light being constant. General relativity describes gravity as the curvature of spacetime caused by mass and energy.",
            contexts=[
                "Albert Einstein proposed the theory of relativity in two parts: special and general relativity.",
                "Special relativity (1905) deals with objects moving at constant speeds and shows that time and space are relative.",
                "General relativity (1915) describes gravity as the curvature of spacetime caused by mass.",
                "The theory predicts that time passes slower in stronger gravitational fields.",
            ],
            metadata={"domain": "physics", "difficulty": "hard"}
        ),
    ]

    return EvaluationDataset("general_knowledge", data_points)


def _create_science_dataset() -> EvaluationDataset:
    """Create a science-focused evaluation dataset."""
    data_points = [
        EvaluationDataPoint(
            question="What is DNA and what is its function?",
            ground_truth="DNA (Deoxyribonucleic acid) is a molecule that carries genetic information in living organisms. It consists of two strands forming a double helix, made up of nucleotides containing bases A, T, G, and C. DNA's function is to store and transmit hereditary information from parents to offspring and to direct protein synthesis.",
            contexts=[
                "DNA is a double-helix molecule composed of nucleotides with four bases: adenine, thymine, guanine, and cytosine.",
                "The primary function of DNA is to store genetic information that determines an organism's characteristics.",
                "DNA directs the synthesis of proteins through the processes of transcription and translation.",
                "DNA is found in the nucleus of eukaryotic cells and contains the instructions for all cellular functions.",
            ],
            metadata={"domain": "biology", "subtopic": "genetics", "difficulty": "medium"}
        ),
        EvaluationDataPoint(
            question="How do vaccines work?",
            ground_truth="Vaccines work by training the immune system to recognize and fight specific pathogens without causing disease. They contain weakened, killed, or parts of disease-causing organisms that trigger immune responses, creating memory cells that provide future protection against the actual pathogen.",
            contexts=[
                "Vaccines contain antigens that stimulate the immune system to produce antibodies.",
                "When exposed to a vaccine, the immune system creates memory cells that remember the pathogen.",
                "If the real pathogen later enters the body, memory cells quickly produce antibodies to fight it.",
                "Different types of vaccines include live attenuated, inactivated, and subunit vaccines.",
            ],
            metadata={"domain": "medicine", "subtopic": "immunology", "difficulty": "medium"}
        ),
    ]

    return EvaluationDataset("science_knowledge", data_points)


def _create_history_dataset() -> EvaluationDataset:
    """Create a history-focused evaluation dataset."""
    data_points = [
        EvaluationDataPoint(
            question="What caused World War I?",
            ground_truth="World War I was caused by a complex mix of factors including imperialism, nationalism, militarism, and alliance systems. The immediate trigger was the assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo on June 28, 1914, which activated the alliance system and led to widespread conflict.",
            contexts=[
                "The assassination of Archduke Franz Ferdinand in Sarajevo sparked the immediate crisis leading to WWI.",
                "European powers had formed complex alliance systems that escalated regional conflicts.",
                "Rising nationalism in the Balkans created tension between Austria-Hungary and Serbia.",
                "Imperial competition and arms races increased tensions between major powers.",
            ],
            metadata={"domain": "history", "period": "early_20th_century", "difficulty": "medium"}
        ),
    ]

    return EvaluationDataset("history_knowledge", data_points)


def _create_technology_dataset() -> EvaluationDataset:
    """Create a technology-focused evaluation dataset."""
    data_points = [
        EvaluationDataPoint(
            question="What is artificial intelligence?",
            ground_truth="Artificial Intelligence (AI) is a field of computer science that aims to create systems capable of performing tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. AI systems use algorithms and data to make decisions and predictions.",
            contexts=[
                "AI is a branch of computer science focused on creating intelligent machines.",
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "AI applications include natural language processing, computer vision, and robotics.",
                "Modern AI systems use neural networks and deep learning algorithms.",
            ],
            metadata={"domain": "technology", "subtopic": "ai", "difficulty": "medium"}
        ),
    ]

    return EvaluationDataset("technology_knowledge", data_points)


def create_custom_dataset(
    questions: List[str],
    ground_truths: List[str],
    contexts_list: Optional[List[List[str]]] = None,
    metadata_list: Optional[List[Dict[str, any]]] = None,
    name: str = "custom_dataset"
) -> EvaluationDataset:
    """Create a custom evaluation dataset from provided data."""
    if len(questions) != len(ground_truths):
        raise ValueError("Number of questions must match number of ground truths")

    data_points = []
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        contexts = contexts_list[i] if contexts_list else []
        metadata = metadata_list[i] if metadata_list else {}

        data_points.append(EvaluationDataPoint(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts,
            metadata=metadata
        ))

    return EvaluationDataset(name, data_points)