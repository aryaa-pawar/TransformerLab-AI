import time
import torch
import pandas as pd

from transformers import pipeline

from src.nlp_tasks import (
    get_sentiment_model,
    get_zero_shot_model,
    get_ner_model,
    get_qa_model
)

from src.generation import (
    get_summarization_model,
    get_text_generator
)


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "Sentiment — DistilBERT": {
        "loader": get_sentiment_model,
        "task": "Sentiment Analysis",
        "architecture": "Encoder-only Transformer"
    },

    "Zero-Shot — BART": {
        "loader": get_zero_shot_model,
        "task": "Zero-Shot Classification",
        "architecture": "Encoder-Decoder Transformer"
    },

    "NER — BERT": {
        "loader": get_ner_model,
        "task": "Named Entity Recognition",
        "architecture": "Encoder-only Transformer"
    },

    "Question Answering — DistilBERT": {
        "loader": get_qa_model,
        "task": "Extractive Question Answering",
        "architecture": "Encoder-only Transformer"
    },

    "Summarization — DistilBART": {
        "loader": get_summarization_model,
        "task": "Text Summarization",
        "architecture": "Encoder-Decoder Transformer"
    },

    "Text Generation — SmolLM2": {
        "loader": get_text_generator,
        "task": "Autoregressive Text Generation",
        "architecture": "Decoder-only Transformer"
    }
}


# ============================================================
# PARAMETER INFORMATION
# ============================================================

def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def get_model_memory_mb(model):

    total_bytes = 0

    for parameter in model.parameters():

        total_bytes += (
            parameter.numel()
            * parameter.element_size()
        )

    for buffer in model.buffers():

        total_bytes += (
            buffer.numel()
            * buffer.element_size()
        )

    return round(
        total_bytes / (1024 ** 2),
        2
    )


# ============================================================
# CONFIG VALUE HELPER
# ============================================================

def get_config_value(
    config,
    names,
    default="N/A"
):

    for name in names:

        value = getattr(
            config,
            name,
            None
        )

        if value is not None:
            return value

    return default


# ============================================================
# TOKENIZER INSPECTOR
# ============================================================

def inspect_tokenizer(
    text,
    model_choice
):

    if not text or not text.strip():

        return {
            "error":
                "Please enter some text."
        }

    if model_choice not in MODEL_REGISTRY:

        return {
            "error":
                "Unknown model selection."
        }

    selected = MODEL_REGISTRY[
        model_choice
    ]

    model_pipeline = (
        selected["loader"]()
    )

    tokenizer = (
        model_pipeline.tokenizer
    )

    model = (
        model_pipeline.model
    )

    encoded = tokenizer(
        text,
        add_special_tokens=True
    )

    token_ids = encoded[
        "input_ids"
    ]

    tokens = (
        tokenizer
        .convert_ids_to_tokens(
            token_ids
        )
    )

    token_data = []

    for position, (
        token,
        token_id
    ) in enumerate(
        zip(
            tokens,
            token_ids
        )
    ):

        token_data.append({
            "Position": position,
            "Token": token,
            "Token ID": int(token_id)
        })

    token_table = pd.DataFrame(
        token_data
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    config = model.config

    hidden_size = get_config_value(
        config,
        [
            "hidden_size",
            "d_model",
            "n_embd",
            "dim"
        ]
    )

    num_layers = get_config_value(
        config,
        [
            "num_hidden_layers",
            "encoder_layers",
            "n_layer",
            "num_layers"
        ]
    )

    attention_heads = get_config_value(
        config,
        [
            "num_attention_heads",
            "encoder_attention_heads",
            "n_head",
            "num_heads"
        ]
    )

    vocab_size = get_config_value(
        config,
        ["vocab_size"],
        len(tokenizer)
    )

    model_name = getattr(
        config,
        "_name_or_path",
        model_choice
    )

    return {
        "task":
            selected["task"],

        "architecture":
            selected["architecture"],

        "model":
            model_name,

        "tokenizer":
            tokenizer.__class__.__name__,

        "characters":
            len(text),

        "words":
            len(text.split()),

        "tokens":
            len(token_ids),

        "vocabulary_size":
            vocab_size,

        "hidden_dimension":
            hidden_size,

        "transformer_layers":
            num_layers,

        "attention_heads":
            attention_heads,

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "approx_memory_mb":
            get_model_memory_mb(
                model
            ),

        "token_table":
            token_table
    }


# ============================================================
# FORMAT MODEL INFORMATION
# ============================================================

def format_model_information(
    result
):

    if "error" in result:
        return result["error"]

    return f"""
TASK
{result['task']}

ARCHITECTURE
{result['architecture']}

MODEL
{result['model']}

TOKENIZER
{result['tokenizer']}

TOKENS IN INPUT
{result['tokens']}

VOCABULARY SIZE
{result['vocabulary_size']}

HIDDEN DIMENSION
{result['hidden_dimension']}

TRANSFORMER LAYERS
{result['transformer_layers']}

ATTENTION HEADS
{result['attention_heads']}

TOTAL PARAMETERS
{result['total_parameters']:,}

TRAINABLE PARAMETERS
{result['trainable_parameters']:,}

APPROXIMATE MODEL MEMORY
{result['approx_memory_mb']} MB
""".strip()


# ============================================================
# TOKENIZER STATISTICS
# ============================================================

def tokenizer_stats(
    text,
    model_choice
):

    if not text:

        return 0, 0, 0

    if model_choice not in MODEL_REGISTRY:

        return 0, 0, 0

    selected = MODEL_REGISTRY[
        model_choice
    ]

    model_pipeline = (
        selected["loader"]()
    )

    tokenizer = (
        model_pipeline.tokenizer
    )

    tokens = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    return (
        len(text),
        len(text.split()),
        len(tokens)
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

COMPARISON_MODEL_NAME = (
    "textattack/bert-base-uncased-SST-2"
)

_comparison_model = None


def get_comparison_model():

    global _comparison_model

    if _comparison_model is None:

        device = (
            0
            if torch.cuda.is_available()
            else -1
        )

        _comparison_model = pipeline(
            "sentiment-analysis",
            model=COMPARISON_MODEL_NAME,
            device=device
        )

    return _comparison_model


# ============================================================
# SINGLE MODEL BENCHMARK
# ============================================================

def benchmark_model(
    model_pipeline,
    text
):

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    result = model_pipeline(
        text
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = (
        time.perf_counter()
        - start
    ) * 1000

    parameters = sum(
        parameter.numel()
        for parameter
        in model_pipeline.model.parameters()
    )

    return {
        "label":
            result["label"],

        "confidence":
            round(
                float(
                    result["score"]
                ) * 100,
                2
            ),

        "latency_ms":
            round(
                latency,
                2
            ),

        "parameters":
            parameters,

        "memory_mb":
            get_model_memory_mb(
                model_pipeline.model
            )
    }


# ============================================================
# COMPARE DISTILBERT AND BERT
# ============================================================

def compare_models(text):

    if not text or not text.strip():

        return {
            "error":
                "Please enter some text."
        }

    distilbert = (
        get_sentiment_model()
    )

    bert = (
        get_comparison_model()
    )

    result_a = benchmark_model(
        distilbert,
        text
    )

    result_b = benchmark_model(
        bert,
        text
    )

    comparison_data = [

        {
            "Model":
                "DistilBERT SST-2",

            "Prediction":
                result_a["label"],

            "Confidence (%)":
                result_a["confidence"],

            "Latency (ms)":
                result_a["latency_ms"],

            "Parameters":
                result_a["parameters"],

            "Approx Memory (MB)":
                result_a["memory_mb"]
        },

        {
            "Model":
                "BERT SST-2",

            "Prediction":
                result_b["label"],

            "Confidence (%)":
                result_b["confidence"],

            "Latency (ms)":
                result_b["latency_ms"],

            "Parameters":
                result_b["parameters"],

            "Approx Memory (MB)":
                result_b["memory_mb"]
        }
    ]

    comparison_table = pd.DataFrame(
        comparison_data
    )

    if (
        result_a["latency_ms"]
        <
        result_b["latency_ms"]
    ):

        faster_model = "DistilBERT"

    else:

        faster_model = "BERT"

    if (
        result_a["confidence"]
        >
        result_b["confidence"]
    ):

        higher_confidence = (
            "DistilBERT"
        )

    elif (
        result_b["confidence"]
        >
        result_a["confidence"]
    ):

        higher_confidence = "BERT"

    else:

        higher_confidence = "Tie"

    agreement = (
        result_a["label"]
        ==
        result_b["label"]
    )

    summary = f"""
Prediction Agreement:
{"Yes" if agreement else "No"}

Faster Model:
{faster_model}

Higher Confidence:
{higher_confidence}

DistilBERT Latency:
{result_a['latency_ms']} ms

BERT Latency:
{result_b['latency_ms']} ms

DistilBERT Parameters:
{result_a['parameters']:,}

BERT Parameters:
{result_b['parameters']:,}

Note:
Runtime latency depends on the current hardware.
Confidence should not be interpreted as overall
model accuracy.
""".strip()

    return {
        "table":
            comparison_table,

        "summary":
            summary
    }
