from enum import auto
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoModelForPreTraining, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


def load_transformer(transformer_name) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a transformer model and tokenizer.

    Args:
        transformer_name (str): The name of the transformer to load.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The transformer model and tokenizer.
    """
    transformer = AutoModelForPreTraining.from_pretrained(
        transformer_name,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    transformer.requires_grad_(False).eval()
    return transformer, AutoTokenizer.from_pretrained(transformer_name)


class EncodeMode(StrEnum):
    """The mode to use for obtaining the sample encoding."""

    MEAN = auto()
    RAW = auto()
    SUM = auto()
    CLS = auto()


@torch.no_grad()
def data_info(attention, start_lemma_index, end_lemma_index, index):
    """Get information from the attention values.

    Args:
        attention (torch.Tensor): The attention values for each layer.
        start_lemma_index (torch.Tensor): The start index of the lemma.
        end_lemma_index (torch.Tensor): The end index of the lemma.
        index (torch.Tensor): The index of the sample.

    Returns:
        Dict[str, List[Any]]: The information.
    """
    result = {}
    for sample_attention, sample_start_lemma_index, sample_end_lemma_index in zip(
        attention, start_lemma_index, end_lemma_index
    ):
        inner_elements = (sample_end_lemma_index - sample_start_lemma_index) ** 2
        outer_elements = sample_attention.shape[1] ** 2 - inner_elements

        inner_attention_mask = torch.zeros_like(sample_attention)
        inner_attention_mask[
            sample_start_lemma_index:sample_end_lemma_index,
            sample_start_lemma_index:sample_end_lemma_index,
        ] = 1

        inner_attention = sample_attention[
            :,
            sample_start_lemma_index:sample_end_lemma_index,
            sample_start_lemma_index:sample_end_lemma_index,
        ]

        attention_mean = {
            f"attention_{i}_mean": layer_attention_mean.cpu().numpy()
            for i, layer_attention_mean in enumerate(sample_attention.mean(dim=[1, 2]))
        }

        inner_attention_mean = {
            f"inner_attention_{i}_mean": inner_attention_mean.cpu().numpy()
            for i, inner_attention_mean in enumerate(inner_attention.mean(dim=[1, 2]))
        }

        outer_attention_mean = {
            f"outer_attention_{i}_mean": outer_attention_mean.cpu().numpy()
            for i, outer_attention_mean in enumerate(
                (sample_attention.sum(dim=[1, 2]) - inner_attention.sum(dim=[1, 2])) / outer_elements
            )
        }

        for k, v in {
            **attention_mean,
            **inner_attention_mean,
            **outer_attention_mean,
        }.items():
            result.setdefault(k, []).append(v)

    return result


@torch.no_grad()
def tokenize(batch, tokenizer: PreTrainedTokenizer, encoder_name: str):
    """Tokenize the batch.

    Args:
        batch (Dict[str, Any]): The batch to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        encoder_name (str): The name of the encoder.

    Returns:
        Dict[str, Any]: The tokenized batch.
    """
    sentence_encoding = tokenizer(
        batch["sentence"],
        return_special_tokens_mask=True,
        return_token_type_ids=False,
        # return_tensors="pt",
        truncation=False,
        max_length=512,
        padding=False,
    )

    sentence_ids = sentence_encoding["input_ids"]
    sentence_special_mask = sentence_encoding["special_tokens_mask"]
    attention_mask = sentence_encoding["attention_mask"]

    lemma_encoding = tokenizer(
        batch["lemma"],
        return_special_tokens_mask=False,
        return_token_type_ids=False,
        # return_tensors="pt",
        truncation=False,
        max_length=512,
        padding=False,
        add_special_tokens=False,
    )

    lemma_ids = lemma_encoding["input_ids"]

    start_lemma_index = []
    end_lemma_index = []
    for sample_index, (sample_sentence_ids, sample_lemma_ids, lemma) in enumerate(
        zip(sentence_ids, lemma_ids, batch["lemma"])
    ):
        matching = []
        for i in range(len(sample_sentence_ids) - len(sample_lemma_ids) + 1):
            if sample_sentence_ids[i : i + len(sample_lemma_ids)] == sample_lemma_ids:
                matching.append((i, i + len(sample_lemma_ids)))

        if len(matching) == 0:
            for prefix in (" ", "-", "."):
                sample_lemma_ids = tokenizer(
                    f"{prefix}{lemma}",
                    return_special_tokens_mask=False,
                    return_token_type_ids=False,
                    # return_tensors="pt",
                    truncation=False,
                    max_length=512,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]
                lemma_ids[sample_index] = sample_lemma_ids

                for i in range(len(sample_sentence_ids) - len(sample_lemma_ids) + 1):
                    if sample_sentence_ids[i : i + len(sample_lemma_ids)] == sample_lemma_ids:
                        matching.append((i, i + len(sample_lemma_ids)))

                if len(matching) != 0:
                    break

        start_lemma, end_lemma = [-1, -1] if len(matching) != 1 else matching[0]
        start_lemma_index.append(start_lemma)
        end_lemma_index.append(end_lemma)

    return {
        "sentence_ids": sentence_ids,
        "sentence_special_mask": sentence_special_mask,
        "attention_mask": attention_mask,
        "lemma_ids": lemma_ids,
        "start_lemma_index": start_lemma_index,
        "end_lemma_index": end_lemma_index,
    }


def dataset_format(dataset: Dataset) -> None:
    """Set the format of the dataset.

    Args:
        dataset (Dataset): The dataset to format."""
    dataset.set_format(
        type="torch",
        columns=[
            column
            for column in dataset.column_names
            if column not in {"synset_id", "lemma", "sentence", "pos", "index"}
        ],
        output_all_columns=True,
    )


class LikelihoodMode(StrEnum):
    """The likelihood mode to use for the perturbation."""

    LOWEST = auto()
    HIGHEST = auto()
    LOW = auto()
    MID = auto()
    HIGH = auto()


@torch.no_grad()
def sample_encode(
    sample,
    index,
    tokenizer: PreTrainedTokenizer,
    encoder: PreTrainedModel,
    encoder_name: str,
    likelihood_mode: LikelihoodMode = None,
    return_tensors: str = "pt",
    # attention_layer: int = -1,
):
    """Encode a sample from a dataset.

    Args:
        sample (dict): A sample from a dataset.
        index (int): The index of the sample in the dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        encoder (PreTrainedModel): The encoder to use.
        encoder_name (str): The name of the encoder.
        likelihood_mode (LikelihoodMode, optional): The likelihood mode to use for the perturbation. Defaults to None.
        return_tensors (str, optional): The format to return the tensors in. Defaults to "pt".

    Returns:
        dict: The encoded sample.
    """
    sentence_ids_key: str = "sentence_ids"
    attention_mask_key: str = "attention_mask"
    start_lemma_index_key: str = "start_lemma_index"
    end_lemma_index_key: str = "end_lemma_index"

    sentence_ids = sample[sentence_ids_key]
    start_lemma_index = sample[start_lemma_index_key]
    end_lemma_index = sample[end_lemma_index_key]

    lemma = sample["lemma"]
    lemma_ids = sentence_ids[start_lemma_index:end_lemma_index]

    if likelihood_mode is not None:
        new_lemma_ids = sample[likelihood_mode]
        lemma = tokenizer.decode(new_lemma_ids)

        assert len(new_lemma_ids) == len(lemma_ids)
        lemma_ids = new_lemma_ids

        sentence_ids[start_lemma_index:end_lemma_index] = new_lemma_ids

    model_out = encoder(
        input_ids=sentence_ids.unsqueeze(0).to(encoder.device),
        attention_mask=sample[attention_mask_key].unsqueeze(0).to(encoder.device),
    )

    attention = torch.stack(
        [layer_attention.squeeze(0).mean(dim=0) for layer_attention in model_out["attentions"]],
        dim=0,
    )

    logits_key: str = "logits" if "logits" in model_out else "prediction_logits"

    likelihoods = model_out[logits_key].squeeze(0).softmax(dim=-1)
    likelihoods = torch.as_tensor([x[token_id] for x, token_id in zip(likelihoods, sentence_ids)])

    inner_likelihoods = likelihoods[start_lemma_index:end_lemma_index]

    outer_likelihoods = torch.cat(
        [
            likelihoods[0:start_lemma_index],
            likelihoods[end_lemma_index:],
        ]
    )
    result = {
        "attention": attention.cpu().numpy(),
        "likelihoods": likelihoods.cpu().numpy(),
        #
        "inner_likelihoods_mean": inner_likelihoods.mean(dim=0).cpu().numpy(),
        "inner_likelihoods_std": inner_likelihoods.std(dim=0).cpu().numpy(),
        "inner_likelihoods_joint": inner_likelihoods.prod(dim=0).cpu().numpy(),
        #
        "outer_likelihoods_mean": outer_likelihoods.mean(dim=0).cpu().numpy(),
        "outer_likelihoods_std": outer_likelihoods.std(dim=0).cpu().numpy(),
        "outer_likelihoods_joint": outer_likelihoods.prod(dim=0).cpu().numpy(),
        #
        "lemma_ids": lemma_ids.cpu().numpy(),
        "lemma": lemma,
    }

    return result


@torch.no_grad()
def perturb(
    sentence_ids,
    attention_mask,
    start_lemma_index,
    end_lemma_index,
    lemma_ids,
    # tokenizer: PreTrainedTokenizer,
    encoder: PreTrainedModel,
    likelihood_modes: Sequence[LikelihoodMode],
    return_tensors: str = "pt",
):
    """Perturb a lemma in a sentence.

    Args:
        sentence_ids (torch.Tensor): The sentence ids.
        attention_mask (torch.Tensor): The attention mask.
        start_lemma_index (int): The start index of the lemma.
        end_lemma_index (int): The end index of the lemma.
        lemma_ids (torch.Tensor): The lemma ids.
        encoder (PreTrainedModel): The encoder to use.
        likelihood_modes (Sequence[LikelihoodMode]): The likelihood modes to use for the perturbation.
        return_tensors (str, optional): The format to return the tensors in. Defaults to "pt".

    Returns:
        dict: The perturbed lemmas with their likelihood mode.
    """
    model_out = encoder(
        input_ids=sentence_ids.unsqueeze(0).to(encoder.device),
        attention_mask=attention_mask.unsqueeze(0).to(encoder.device),
    )
    logits_key: str = "logits" if "logits" in model_out else "prediction_logits"

    likelihoods = model_out[logits_key].squeeze(0).softmax(dim=-1)
    vocab_size = likelihoods.size(1)

    inner_likelihoods = likelihoods[start_lemma_index:end_lemma_index]
    likelihood_mode2lemma_ids = {}

    for likelihood_mode in likelihood_modes:
        if likelihood_mode == LikelihoodMode.LOWEST:
            portion = [0, 1]
        elif likelihood_mode == LikelihoodMode.HIGHEST:
            portion = [vocab_size - 1, vocab_size]
        elif likelihood_mode in {
            LikelihoodMode.LOW,
            LikelihoodMode.MID,
            LikelihoodMode.HIGH,
        }:
            if likelihood_mode == LikelihoodMode.LOW:
                portion = [1, vocab_size // 3]
            elif likelihood_mode == LikelihoodMode.MID:
                portion = [vocab_size // 3, vocab_size // 3 * 2]
            elif likelihood_mode == LikelihoodMode.HIGH:
                portion = [vocab_size // 3 * 2, vocab_size - 1]
            else:
                raise ValueError(f"Unknown likelihood mode {likelihood_mode}")
        else:
            raise ValueError(f"Unknown likelihood mode {likelihood_mode}")

        _, sort_indices = inner_likelihoods.sort(dim=1, descending=False)

        torch.manual_seed(42)
        # sample a random lemma id for each token from the inner_likelihoods
        new_lemma_ids = torch.randint(portion[0], portion[1], (len(lemma_ids),))
        new_lemma_ids = sort_indices.cpu().gather(1, new_lemma_ids.unsqueeze(1)).squeeze(1)
        likelihood_mode2lemma_ids[likelihood_mode] = new_lemma_ids.cpu().numpy()

    return {
        **likelihood_mode2lemma_ids,
        "sentence_ids": sentence_ids.cpu().numpy(),
    }


def load_data(data_path: Path):
    """Load the data.

    Args:
        data_path (Path): The path to the data.

    Returns:
        DatasetDict: The data."""
    data = DatasetDict.load_from_disk(data_path)
    for _, encoder_data in data.items():
        encoder_data.set_format(
            type="torch",
            columns=[
                column
                for column in encoder_data.column_names
                if column not in {"synset_id", "lemma", "sentence", "pos", "index"}
            ],
            output_all_columns=True,
        )
    return data


def data_to_df(data: DatasetDict):
    """Convert the data to a dataframe.

    Args:
        data (DatasetDict): The data.

    Returns:
        pd.DataFrame: The dataframe."""
    df = {
        "encoder": [],
        "layer": [],
        "attention": [],
        "inner_attention": [],
        "outer_attention": [],
        "likelihood": [],
    }
    for encoder_name, encoder_data in data.items():
        for layer in range(50):
            if f"attention_{layer}_mean" not in encoder_data.column_names:
                continue

            df["encoder"].extend([encoder_name] * len(encoder_data))
            df["layer"].extend([layer] * len(encoder_data))
            df["attention"].extend(encoder_data[f"attention_{layer}_mean"].squeeze(1).tolist())
            df["inner_attention"].extend(encoder_data[f"inner_attention_{layer}_mean"].squeeze(1).tolist())
            df["outer_attention"].extend(encoder_data[f"outer_attention_{layer}_mean"].squeeze(1).tolist())
            # df["likelihood_mean"].extend(encoder_data.filter()["inner_likelihoods_mean"].squeeze(1).tolist())
            # df["likelihood_joint"].extend(encoder_data.filter()["inner_likelihoods_joint"].squeeze(1).tolist())
            df["likelihood"].extend(encoder_data["inner_likelihoods_mean"].squeeze(1).tolist())

    df = pd.DataFrame(df)

    return df
