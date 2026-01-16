import numpy as np
import torch

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys, collate_fn_to_dict_list


class DummyTokenizer:
    """Minimal tokenizer stub that behaves like a HF tokenizer for padding."""

    def __init__(self, pad_token_id: int = 0, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.model_input_names = ["input_ids", "attention_mask", "labels"]

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        assert padding in [True, "max_length"]
        target_length = max_length or max(len(feature["input_ids"]) for feature in encoded_inputs)
        pad_values = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        padded = {key: [] for key in encoded_inputs[0].keys()}
        for feature in encoded_inputs:
            for key, value in feature.items():
                value_list = list(value)
                pad_value = pad_values.get(key, 0)
                padded[key].append(value_list + [pad_value] * (target_length - len(value_list)))
        if return_tensors == "pt":
            for key in padded:
                padded[key] = torch.tensor(padded[key], dtype=torch.long)
        return padded


def test_collate_fn_to_dict_list_merges_tensor_and_python_data():
    data_list = [
        {"input_ids": torch.tensor([[1, 2]]), "meta": {"id": "a"}},
        {"input_ids": torch.tensor([[3, 4]]), "meta": {"id": "b"}},
    ]

    output = collate_fn_to_dict_list(data_list)

    assert torch.equal(output["input_ids"], torch.tensor([[1, 2], [3, 4]]))
    assert isinstance(output["meta"], np.ndarray)
    assert output["meta"].shape == (2,)
    assert output["meta"][0]["id"] == "a"
    assert output["meta"][1]["id"] == "b"


def test_data_collator_with_padding_for_padded_keys_handles_unpadded_fields():
    tokenizer = DummyTokenizer(pad_token_id=9)
    collator = DataCollatorWithPaddingForPaddedKeys(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=6,
    )

    features = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [10, 11, 12],
            "auxiliary": {"type": 1},
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [13, 14],
            "auxiliary": {"type": 2},
        },
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == (2, 6)
    assert torch.equal(batch["input_ids"][0, 3:], torch.tensor([9, 9, 9]))
    assert torch.equal(batch["attention_mask"][1], torch.tensor([1, 1, 0, 0, 0, 0]))
    assert torch.equal(batch["labels"][1], torch.tensor([13, 14, -100, -100, -100, -100]))

    expected_position_ids = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2],
            [0, 1, 1, 1, 1, 1],
        ]
    )
    assert torch.equal(batch["position_ids"], expected_position_ids)

    assert isinstance(batch["auxiliary"], np.ndarray)
    assert batch["auxiliary"][0]["type"] == 1
    assert batch["auxiliary"][1]["type"] == 2
