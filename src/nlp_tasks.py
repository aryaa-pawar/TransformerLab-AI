import torch
from transformers import pipeline


# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else -1


# ============================================================
# MODEL NAMES
# ============================================================

SENTIMENT_MODEL = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

NER_MODEL = "dslim/bert-base-NER"

QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


# ============================================================
# LAZY MODEL STORAGE
# Models load only when their task is first used.
# ============================================================

_models = {}


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def get_sentiment_model():

    if "sentiment" not in _models:

        _models["sentiment"] = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=DEVICE
        )

    return _models["sentiment"]


def analyze_sentiment(text):

    if not text or not text.strip():
        return {
            "error": "Please enter some text."
        }

    model = get_sentiment_model()

    result = model(text)[0]

    return {
        "task": "Sentiment Analysis",
        "label": result["label"],
        "confidence": round(
            float(result["score"]) * 100,
            2
        ),
        "model": "DistilBERT SST-2"
    }


# ============================================================
# ZERO-SHOT CLASSIFICATION
# ============================================================

def get_zero_shot_model():

    if "zero_shot" not in _models:

        _models["zero_shot"] = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL,
            device=DEVICE
        )

    return _models["zero_shot"]


def classify_zero_shot(text, labels):

    if not text or not text.strip():
        return {
            "error": "Please enter some text."
        }

    if not labels:
        return {
            "error": "Please provide candidate labels."
        }

    if isinstance(labels, str):

        labels = [
            label.strip()
            for label in labels.split(",")
            if label.strip()
        ]

    if not labels:
        return {
            "error": "Please provide candidate labels."
        }

    model = get_zero_shot_model()

    result = model(
        text,
        candidate_labels=labels
    )

    predictions = []

    for label, score in zip(
        result["labels"],
        result["scores"]
    ):

        predictions.append({
            "label": label,
            "confidence": round(
                float(score) * 100,
                2
            )
        })

    return {
        "task": "Zero-Shot Classification",
        "predictions": predictions,
        "model": "BART Large MNLI"
    }


# ============================================================
# NAMED ENTITY RECOGNITION
# ============================================================

def get_ner_model():

    if "ner" not in _models:

        _models["ner"] = pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
            device=DEVICE
        )

    return _models["ner"]


def analyze_entities(text):

    if not text or not text.strip():
        return {
            "error": "Please enter some text."
        }

    model = get_ner_model()

    results = model(text)

    entities = []

    for entity in results:

        entities.append({
            "text": entity["word"],
            "type": entity["entity_group"],
            "confidence": round(
                float(entity["score"]) * 100,
                2
            ),
            "start": int(entity["start"]),
            "end": int(entity["end"])
        })

    return {
        "task": "Named Entity Recognition",
        "entities": entities,
        "model": "BERT Base NER"
    }


# ============================================================
# EXTRACTIVE QUESTION ANSWERING
# ============================================================

def get_qa_model():

    if "qa" not in _models:

        _models["qa"] = pipeline(
            "question-answering",
            model=QA_MODEL,
            device=DEVICE
        )

    return _models["qa"]


def answer_question(question, context):

    if not question or not question.strip():
        return {
            "error": "Please enter a question."
        }

    if not context or not context.strip():
        return {
            "error": "Please provide some context."
        }

    model = get_qa_model()

    result = model(
        question=question,
        context=context
    )

    return {
        "task": "Extractive Question Answering",
        "answer": result["answer"],
        "confidence": round(
            float(result["score"]) * 100,
            2
        ),
        "start": int(result["start"]),
        "end": int(result["end"]),
        "model": "DistilBERT SQuAD"
    }


def highlight_answer(context, start, end):

    before = context[:start]
    answer = context[start:end]
    after = context[end:]

    return (
        before
        + "**"
        + answer
        + "**"
        + after
    )
