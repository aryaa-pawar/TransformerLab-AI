import time
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL NAMES
# ============================================================

SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

GENERATION_MODEL = "HuggingFaceTB/SmolLM2-360M"

TRANSLATION_MODELS = {
    "English → French": "Helsinki-NLP/opus-mt-en-fr",
    "English → German": "Helsinki-NLP/opus-mt-en-de",
    "English → Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "French → English": "Helsinki-NLP/opus-mt-fr-en",
    "German → English": "Helsinki-NLP/opus-mt-de-en"
}


# ============================================================
# LAZY MODEL STORAGE
# ============================================================

_models = {}
_translation_cache = {}


# ============================================================
# SUMMARIZATION
# ============================================================

def get_summarization_model():

    if "summarization" not in _models:

        _models["summarization"] = pipeline(
            "summarization",
            model=SUMMARIZATION_MODEL,
            device=DEVICE
        )

    return _models["summarization"]


def summarize_text(text, length="Medium"):

    if not text or not text.strip():

        return {
            "error": "Please enter some text."
        }

    word_count = len(text.split())

    if word_count < 40:

        return {
            "error":
                "Please provide at least 40 words "
                "for summarization."
        }

    length_settings = {

        "Short": {
            "max_length": 60,
            "min_length": 20
        },

        "Medium": {
            "max_length": 120,
            "min_length": 40
        },

        "Detailed": {
            "max_length": 180,
            "min_length": 70
        }
    }

    settings = length_settings.get(
        length,
        length_settings["Medium"]
    )

    model = get_summarization_model()

    result = model(
        text,
        max_length=settings["max_length"],
        min_length=settings["min_length"],
        do_sample=False,
        truncation=True
    )

    summary = result[0]["summary_text"]

    original_words = len(text.split())
    summary_words = len(summary.split())

    compression = round(
        (
            1 -
            (summary_words / original_words)
        ) * 100,
        2
    )

    return {
        "task": "Text Summarization",
        "summary": summary,
        "original_words": original_words,
        "summary_words": summary_words,
        "compression": compression,
        "model": "DistilBART CNN"
    }


# ============================================================
# TRANSLATION
# ============================================================

def get_translation_model(language_pair):

    if language_pair not in TRANSLATION_MODELS:

        raise ValueError(
            f"Unsupported language pair: "
            f"{language_pair}"
        )

    if language_pair not in _translation_cache:

        model_name = TRANSLATION_MODELS[
            language_pair
        ]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(model_name)
        )

        model = model.to(
            TORCH_DEVICE
        )

        model.eval()

        _translation_cache[
            language_pair
        ] = {
            "tokenizer": tokenizer,
            "model": model
        }

    return _translation_cache[
        language_pair
    ]


def translate_text(
    text,
    language_pair
):

    if not text or not text.strip():

        return {
            "error":
                "Please enter text to translate."
        }

    try:

        translator = get_translation_model(
            language_pair
        )

        tokenizer = translator[
            "tokenizer"
        ]

        model = translator[
            "model"
        ]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(TORCH_DEVICE)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True
            )

        translated_text = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )

        return {
            "task": "Machine Translation",
            "translation": translated_text,
            "language_pair": language_pair,
            "model":
                TRANSLATION_MODELS[
                    language_pair
                ],
            "input_tokens":
                int(
                    inputs[
                        "input_ids"
                    ].shape[1]
                ),
            "output_tokens":
                int(
                    generated_tokens.shape[1]
                )
        }

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# TEXT GENERATION
# ============================================================

def get_text_generator():

    if "generation" not in _models:

        _models["generation"] = pipeline(
            "text-generation",
            model=GENERATION_MODEL,
            device=DEVICE
        )

    return _models["generation"]


def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    num_sequences=1
):

    if not prompt or not prompt.strip():

        return {
            "error":
                "Please enter a prompt."
        }

    temperature = float(
        temperature
    )

    top_p = float(
        top_p
    )

    max_new_tokens = int(
        max_new_tokens
    )

    num_sequences = int(
        num_sequences
    )

    generator = get_text_generator()

    start_time = time.time()

    results = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_sequences,
        pad_token_id=(
            generator
            .tokenizer
            .eos_token_id
        )
    )

    inference_time = (
        time.time() - start_time
    ) * 1000

    generations = []

    for index, result in enumerate(
        results
    ):

        full_text = result[
            "generated_text"
        ]

        if full_text.startswith(
            prompt
        ):

            generated_part = (
                full_text[
                    len(prompt):
                ].strip()
            )

        else:

            generated_part = (
                full_text.strip()
            )

        generations.append({
            "sequence": index + 1,
            "text": generated_part
        })

    input_tokens = (
        generator
        .tokenizer
        .encode(prompt)
    )

    return {
        "task": "Text Generation",
        "generations": generations,
        "model": GENERATION_MODEL,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens":
            max_new_tokens,
        "input_tokens":
            len(input_tokens),
        "inference_time_ms":
            round(
                inference_time,
                2
            )
    }
