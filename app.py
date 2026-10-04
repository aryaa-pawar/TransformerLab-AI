import gradio as gr

from src.nlp_tasks import (
    analyze_sentiment,
    classify_zero_shot,
    analyze_entities,
    answer_question,
    highlight_answer
)

from src.generation import (
    summarize_text,
    translate_text,
    generate_text,
    TRANSLATION_MODELS
)

from src.semantic_search import (
    semantic_search,
    format_semantic_results,
    EMBEDDING_MODEL
)

from src.document_processor import (
    search_document,
    format_document_results
)

from src.model_insights import (
    MODEL_REGISTRY,
    inspect_tokenizer,
    format_model_information,
    tokenizer_stats,
    compare_models
)


# ============================================================
# NLP WORKBENCH WRAPPERS
# ============================================================

def ui_sentiment(text):

    result = analyze_sentiment(text)

    if "error" in result:
        return result["error"]

    return (
        f"Prediction: {result['label']}\n\n"
        f"Confidence: {result['confidence']}%\n\n"
        f"Model: {result['model']}"
    )


def ui_zero_shot(text, labels):

    label_list = [
        label.strip()
        for label in labels.split(",")
        if label.strip()
    ]

    result = classify_zero_shot(
        text,
        label_list
    )

    if "error" in result:
        return result["error"]

    lines = []

    for item in result["predictions"]:

        lines.append(
            f"{item['label']}: "
            f"{item['confidence']}%"
        )

    return "\n".join(lines)


def ui_ner(text):

    result = analyze_entities(text)

    if "error" in result:
        return result["error"]

    if not result["entities"]:
        return "No named entities detected."

    lines = []

    for entity in result["entities"]:

        lines.append(
            f"{entity['text']}  →  "
            f"{entity['type']}  "
            f"({entity['confidence']}%)"
        )

    return "\n".join(lines)


def ui_qa(context, question):

    result = answer_question(
        question,
        context
    )

    if "error" in result:
        return result["error"], ""

    answer = (
        f"Answer: {result['answer']}\n\n"
        f"Confidence: "
        f"{result['confidence']}%\n\n"
        f"Model: {result['model']}"
    )

    evidence = highlight_answer(
        context,
        result["start"],
        result["end"]
    )

    return answer, evidence


# ============================================================
# GENERATIVE NLP WRAPPERS
# ============================================================

def ui_summary(text, length):

    result = summarize_text(
        text,
        length
    )

    if "error" in result:
        return result["error"], ""

    stats = (
        f"Original words: "
        f"{result['original_words']}\n\n"

        f"Summary words: "
        f"{result['summary_words']}\n\n"

        f"Compression: "
        f"{result['compression']}%\n\n"

        f"Model: {result['model']}"
    )

    return (
        result["summary"],
        stats
    )


def ui_translation(
    text,
    language_pair
):

    result = translate_text(
        text,
        language_pair
    )

    if "error" in result:
        return result["error"], ""

    info = (
        f"Language Pair: "
        f"{result['language_pair']}\n\n"

        f"Model: "
        f"{result['model']}\n\n"

        f"Input Tokens: "
        f"{result['input_tokens']}\n\n"

        f"Output Tokens: "
        f"{result['output_tokens']}"
    )

    return (
        result["translation"],
        info
    )


def ui_generation(
    prompt,
    max_tokens,
    temperature,
    top_p,
    sequences
):

    result = generate_text(
        prompt,
        max_tokens,
        temperature,
        top_p,
        sequences
    )

    if "error" in result:
        return result["error"], ""

    outputs = []

    for generation in result[
        "generations"
    ]:

        outputs.append(
            f"GENERATION "
            f"{generation['sequence']}\n\n"
            f"{generation['text']}"
        )

    generated_text = (
        "\n\n"
        + "-" * 55
        + "\n\n"
    ).join(outputs)

    info = (
        f"Model: {result['model']}\n\n"
        f"Temperature: "
        f"{result['temperature']}\n\n"
        f"Top-p: {result['top_p']}\n\n"
        f"Input Tokens: "
        f"{result['input_tokens']}\n\n"
        f"Inference Time: "
        f"{result['inference_time_ms']} ms"
    )

    return generated_text, info


# ============================================================
# SEMANTIC SEARCH WRAPPER
# ============================================================

def ui_semantic_search(
    document,
    query,
    top_k
):

    result = semantic_search(
        document,
        query,
        top_k
    )

    if "error" in result:
        return result["error"], ""

    output = format_semantic_results(
        result
    )

    info = (
        f"Embedding Model:\n"
        f"{result['model']}\n\n"

        f"Document Chunks: "
        f"{result['total_chunks']}\n\n"

        f"Embedding Dimension: "
        f"{result['embedding_dimension']}\n\n"

        f"Search Time: "
        f"{result['search_time_ms']} ms"
    )

    return output, info


# ============================================================
# DOCUMENT INTELLIGENCE WRAPPER
# ============================================================

def ui_document_search(
    file_path,
    query,
    top_k
):

    result = search_document(
        file_path,
        query,
        top_k
    )

    if "error" in result:
        return result["error"], ""

    output = format_document_results(
        result
    )

    info = (
        f"File Type: "
        f"{result['file_type']}\n\n"

        f"Pages Processed: "
        f"{result['pages']}\n\n"

        f"Words: "
        f"{result['word_count']}\n\n"

        f"Chunks: "
        f"{result['chunks']}\n\n"

        f"Embedding Model:\n"
        f"{result['model']}\n\n"

        f"Search Time: "
        f"{result['search_time_ms']} ms"
    )

    return output, info


# ============================================================
# MODEL INSIGHTS WRAPPER
# ============================================================

def ui_insights(
    text,
    model_choice
):

    result = inspect_tokenizer(
        text,
        model_choice
    )

    if "error" in result:
        return None, result["error"]

    return (
        result["token_table"],
        format_model_information(
            result
        )
    )


def ui_comparison(text):

    result = compare_models(text)

    if "error" in result:
        return None, result["error"]

    return (
        result["table"],
        result["summary"]
    )


# ============================================================
# CUSTOM CSS
# ============================================================

CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

.hero {
    text-align: center;
    padding: 24px 10px 14px 10px;
}

.hero h1 {
    font-size: 2.8rem;
    margin-bottom: 8px;
}

.hero p {
    font-size: 1.05rem;
    opacity: 0.82;
}

.section-title {
    margin-top: 8px;
    margin-bottom: 14px;
}

.footer {
    text-align: center;
    opacity: 0.75;
    margin-top: 25px;
}
"""


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(
    title="TransformerLab AI",
    css=CSS
) as app:

    gr.HTML(
        """
        <div class="hero">
            <h1>⚡ TransformerLab AI</h1>

            <p>
            Interactive Transformer & NLP
            Intelligence Workbench
            </p>

            <p>
            Classification • Extraction • Generation •
            Semantic Retrieval • Model Intelligence
            </p>
        </div>
        """
    )

    with gr.Tabs():

        # ====================================================
        # NLP WORKBENCH
        # ====================================================

        with gr.Tab(
            "🧠 NLP Workbench"
        ):

            gr.Markdown(
                """
## NLP Understanding

Run Transformer models for classification,
entity extraction and question answering.
                """
            )

            with gr.Tabs():

                # SENTIMENT

                with gr.Tab(
                    "😊 Sentiment"
                ):

                    sentiment_input = gr.Textbox(
                        lines=7,
                        label="Input Text",
                        placeholder=(
                            "Enter text to analyze..."
                        )
                    )

                    sentiment_button = gr.Button(
                        "Analyze Sentiment",
                        variant="primary"
                    )

                    sentiment_output = gr.Textbox(
                        lines=7,
                        label="Analysis"
                    )

                    sentiment_button.click(
                        ui_sentiment,
                        sentiment_input,
                        sentiment_output
                    )


                # ZERO SHOT

                with gr.Tab(
                    "🏷️ Zero-Shot"
                ):

                    zero_text = gr.Textbox(
                        lines=7,
                        label="Input Text"
                    )

                    zero_labels = gr.Textbox(
                        label="Candidate Labels",
                        placeholder=(
                            "technology, finance, "
                            "sports, healthcare"
                        )
                    )

                    zero_button = gr.Button(
                        "Classify Text",
                        variant="primary"
                    )

                    zero_output = gr.Textbox(
                        lines=10,
                        label="Ranked Categories"
                    )

                    zero_button.click(
                        ui_zero_shot,
                        [
                            zero_text,
                            zero_labels
                        ],
                        zero_output
                    )


                # NER

                with gr.Tab(
                    "👤 Named Entities"
                ):

                    ner_input = gr.Textbox(
                        lines=8,
                        label="Input Text",
                        placeholder=(
                            "Example: Sundar Pichai "
                            "works at Google."
                        )
                    )

                    ner_button = gr.Button(
                        "Detect Entities",
                        variant="primary"
                    )

                    ner_output = gr.Textbox(
                        lines=12,
                        label="Detected Entities"
                    )

                    ner_button.click(
                        ui_ner,
                        ner_input,
                        ner_output
                    )


                # QA

                with gr.Tab(
                    "❓ Question Answering"
                ):

                    qa_context = gr.Textbox(
                        lines=12,
                        label="Context"
                    )

                    qa_question = gr.Textbox(
                        label="Question"
                    )

                    qa_button = gr.Button(
                        "Find Answer",
                        variant="primary"
                    )

                    qa_answer = gr.Textbox(
                        lines=6,
                        label="Extracted Answer"
                    )

                    qa_evidence = gr.Markdown(
                        label="Evidence"
                    )

                    qa_button.click(
                        ui_qa,
                        [
                            qa_context,
                            qa_question
                        ],
                        [
                            qa_answer,
                            qa_evidence
                        ]
                    )


        # ====================================================
        # GENERATIVE NLP
        # ====================================================

        with gr.Tab(
            "✨ Generative NLP"
        ):

            gr.Markdown(
                """
## Generative Transformers

Explore summarization, translation and
autoregressive language generation.
                """
            )

            with gr.Tabs():

                # SUMMARY

                with gr.Tab(
                    "📝 Summarization"
                ):

                    summary_input = gr.Textbox(
                        lines=14,
                        label="Input Document"
                    )

                    summary_length = gr.Radio(
                        [
                            "Short",
                            "Medium",
                            "Detailed"
                        ],
                        value="Medium",
                        label="Summary Length"
                    )

                    summary_button = gr.Button(
                        "Generate Summary",
                        variant="primary"
                    )

                    with gr.Row():

                        summary_output = gr.Textbox(
                            lines=12,
                            label="Summary"
                        )

                        summary_stats = gr.Textbox(
                            lines=12,
                            label="Statistics"
                        )

                    summary_button.click(
                        ui_summary,
                        [
                            summary_input,
                            summary_length
                        ],
                        [
                            summary_output,
                            summary_stats
                        ]
                    )


                # TRANSLATION

                with gr.Tab(
                    "🌍 Translation"
                ):

                    translation_input = (
                        gr.Textbox(
                            lines=10,
                            label="Input Text"
                        )
                    )

                    translation_pair = (
                        gr.Dropdown(
                            choices=list(
                                TRANSLATION_MODELS
                                .keys()
                            ),
                            value=(
                                "English → French"
                            ),
                            label="Language Pair"
                        )
                    )

                    translation_button = (
                        gr.Button(
                            "Translate",
                            variant="primary"
                        )
                    )

                    with gr.Row():

                        translation_output = (
                            gr.Textbox(
                                lines=10,
                                label="Translation"
                            )
                        )

                        translation_info = (
                            gr.Textbox(
                                lines=10,
                                label="Model Info"
                            )
                        )

                    translation_button.click(
                        ui_translation,
                        [
                            translation_input,
                            translation_pair
                        ],
                        [
                            translation_output,
                            translation_info
                        ]
                    )


                # GENERATION

                with gr.Tab(
                    "✍️ Text Generation"
                ):

                    generation_prompt = (
                        gr.Textbox(
                            lines=7,
                            label="Prompt"
                        )
                    )

                    with gr.Row():

                        generation_tokens = (
                            gr.Slider(
                                20,
                                250,
                                value=100,
                                step=10,
                                label=(
                                    "Max New Tokens"
                                )
                            )
                        )

                        generation_temp = (
                            gr.Slider(
                                0.1,
                                1.5,
                                value=0.7,
                                step=0.1,
                                label="Temperature"
                            )
                        )

                    with gr.Row():

                        generation_top_p = (
                            gr.Slider(
                                0.1,
                                1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p"
                            )
                        )

                        generation_sequences = (
                            gr.Slider(
                                1,
                                3,
                                value=1,
                                step=1,
                                label="Sequences"
                            )
                        )

                    generation_button = (
                        gr.Button(
                            "Generate Text",
                            variant="primary"
                        )
                    )

                    with gr.Row():

                        generation_output = (
                            gr.Textbox(
                                lines=16,
                                label=(
                                    "Generated Text"
                                )
                            )
                        )

                        generation_info = (
                            gr.Textbox(
                                lines=16,
                                label=(
                                    "Generation Info"
                                )
                            )
                        )

                    generation_button.click(
                        ui_generation,
                        [
                            generation_prompt,
                            generation_tokens,
                            generation_temp,
                            generation_top_p,
                            generation_sequences
                        ],
                        [
                            generation_output,
                            generation_info
                        ]
                    )


        # ====================================================
        # SEMANTIC SEARCH
        # ====================================================

        with gr.Tab(
            "🔎 Semantic Search"
        ):

            gr.Markdown(
                """
## Search by Meaning

Retrieve passages using MiniLM embeddings
and cosine similarity rather than exact
keyword matching.
                """
            )

            semantic_document = gr.Textbox(
                lines=17,
                label="Document"
            )

            semantic_query = gr.Textbox(
                label="Search Query"
            )

            semantic_top_k = gr.Slider(
                1,
                5,
                value=3,
                step=1,
                label="Top Results"
            )

            semantic_button = gr.Button(
                "Search Document",
                variant="primary"
            )

            with gr.Row():

                semantic_output = gr.Textbox(
                    lines=18,
                    label="Relevant Passages"
                )

                semantic_info = gr.Textbox(
                    lines=18,
                    label="Retrieval Info"
                )

            semantic_button.click(
                ui_semantic_search,
                [
                    semantic_document,
                    semantic_query,
                    semantic_top_k
                ],
                [
                    semantic_output,
                    semantic_info
                ]
            )


        # ====================================================
        # DOCUMENT INTELLIGENCE
        # ====================================================

        with gr.Tab(
            "📄 Document Intelligence"
        ):

            gr.Markdown(
                """
## PDF / TXT Semantic Retrieval

Upload a text-based PDF or TXT document
and retrieve the most semantically relevant
passages with page-aware results.

Scanned PDFs requiring OCR are not supported
in this version.
                """
            )

            document_file = gr.File(
                label="Upload PDF / TXT",
                file_types=[
                    ".pdf",
                    ".txt"
                ],
                type="filepath"
            )

            document_query = gr.Textbox(
                label="Search Query",
                placeholder=(
                    "What information "
                    "are you looking for?"
                )
            )

            document_top_k = gr.Slider(
                1,
                5,
                value=3,
                step=1,
                label="Top Results"
            )

            document_button = gr.Button(
                "Search Uploaded Document",
                variant="primary"
            )

            with gr.Row():

                document_output = gr.Textbox(
                    lines=20,
                    label="Relevant Passages"
                )

                document_info = gr.Textbox(
                    lines=20,
                    label="Document Info"
                )

            document_button.click(
                ui_document_search,
                [
                    document_file,
                    document_query,
                    document_top_k
                ],
                [
                    document_output,
                    document_info
                ]
            )


        # ====================================================
        # MODEL LAB
        # ====================================================

        with gr.Tab(
            "🧪 Model Lab"
        ):

            gr.Markdown(
                """
## Transformer Model Intelligence

Inspect tokenization and architecture,
then compare two Transformer models
performing the same sentiment task.
                """
            )

            with gr.Tabs():

                # MODEL INSIGHTS

                with gr.Tab(
                    "🔬 Model Insights"
                ):

                    insight_model = (
                        gr.Dropdown(
                            choices=list(
                                MODEL_REGISTRY
                                .keys()
                            ),
                            value=(
                                "Sentiment — "
                                "DistilBERT"
                            ),
                            label=(
                                "Transformer Model"
                            )
                        )
                    )

                    insight_text = gr.Textbox(
                        lines=7,
                        label="Text to Inspect"
                    )

                    insight_button = gr.Button(
                        "Inspect Transformer",
                        variant="primary"
                    )

                    with gr.Row():

                        insight_chars = gr.Number(
                            label="Characters"
                        )

                        insight_words = gr.Number(
                            label="Words"
                        )

                        insight_tokens = gr.Number(
                            label="Tokens"
                        )

                    insight_table = gr.Dataframe(
                        label="Tokenizer Output",
                        interactive=False
                    )

                    insight_info = gr.Textbox(
                        lines=20,
                        label="Architecture Information"
                    )

                    insight_button.click(
                        ui_insights,
                        [
                            insight_text,
                            insight_model
                        ],
                        [
                            insight_table,
                            insight_info
                        ]
                    )

                    insight_text.change(
                        tokenizer_stats,
                        [
                            insight_text,
                            insight_model
                        ],
                        [
                            insight_chars,
                            insight_words,
                            insight_tokens
                        ]
                    )

                    insight_model.change(
                        tokenizer_stats,
                        [
                            insight_text,
                            insight_model
                        ],
                        [
                            insight_chars,
                            insight_words,
                            insight_tokens
                        ]
                    )


                # MODEL COMPARISON

                with gr.Tab(
                    "⚖️ Model Comparison"
                ):

                    gr.Markdown(
                        """
Compare **DistilBERT** and **BERT**
on the same sentiment input.

Metrics include prediction confidence,
runtime latency, parameter count and
approximate model memory.
                        """
                    )

                    comparison_input = (
                        gr.Textbox(
                            lines=7,
                            label=(
                                "Sentiment Text"
                            )
                        )
                    )

                    comparison_button = (
                        gr.Button(
                            "Compare Models",
                            variant="primary"
                        )
                    )

                    comparison_table = (
                        gr.Dataframe(
                            label=(
                                "Benchmark Results"
                            ),
                            interactive=False
                        )
                    )

                    comparison_summary = (
                        gr.Textbox(
                            lines=16,
                            label=(
                                "Comparison Summary"
                            )
                        )
                    )

                    comparison_button.click(
                        ui_comparison,
                        comparison_input,
                        [
                            comparison_table,
                            comparison_summary
                        ]
                    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

        <hr>

        <strong>TransformerLab AI</strong><br>

        Built with PyTorch • Hugging Face Transformers •
        Sentence Transformers • Gradio

        </div>
        """
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.launch()
