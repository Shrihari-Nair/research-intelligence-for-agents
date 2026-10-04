import pytest

from research_os.common import load_intents
from research_os.router.data.load import load_split, to_dataset


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A quickly trained SetFit router (5 examples/intent, 1 iteration) saved to disk."""
    from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

    intents = load_intents()
    train = to_dataset(load_split("train"), intents)
    train = sample_dataset(train, label_column="label", num_samples=5, seed=42)
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", labels=intents)
    args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=1, seed=42)
    Trainer(model=model, args=args, train_dataset=train).train()
    path = tmp_path_factory.mktemp("tiny") / "model"
    model.save_pretrained(str(path))
    return str(path)
