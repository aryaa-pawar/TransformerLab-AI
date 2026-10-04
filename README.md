# TransformerLab AI

> An interactive NLP workbench I built while exploring and learning how Transformer models work across different Natural Language Processing tasks.

## About This Project

TransformerLab AI started as a learning and experimentation project.

Instead of building another application around a single AI model, I wanted to try different Transformer architectures myself and understand how they are used for different NLP problems.

The main goal was not to train a huge model from scratch or build a production system. It was to work hands-on with pretrained Transformer models, connect them to real NLP tasks, inspect what happens inside them, and understand the differences between architectures such as BERT, DistilBERT, BART, MarianMT, Sentence Transformers, and decoder-based language models.

While building the project, I experimented with:

- Sentiment Analysis
- Zero-Shot Classification
- Named Entity Recognition
- Extractive Question Answering
- Text Summarization
- Machine Translation
- Text Generation
- Sentence Embeddings
- Semantic Search
- PDF/TXT Document Processing
- Tokenization
- Transformer Architecture Inspection
- Model Comparison

The final result is **TransformerLab AI**, a Gradio-based workbench where these experiments can be explored from one interface.

---

# What I Wanted to Learn

The project was mainly about understanding the Transformer ecosystem by actually using it.

Some of the questions I wanted to explore were:

- How can the same Transformer architecture be adapted for different NLP tasks?
- What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?
- How does tokenization convert text into something a Transformer can process?
- What are token IDs?
- How are pretrained Hugging Face models loaded and used for inference?
- How does zero-shot classification work without training a classifier for my own labels?
- How does extractive question answering find an answer inside a given context?
- How does summarization differ from text generation?
- How do translation models generate text in another language?
- What are sentence embeddings?
- How can semantic similarity be used for search?
- Why can semantic search find relevant information even when the query and document use different words?
- How can documents be divided into chunks before embedding them?
- How do model size and parameter count differ between Transformer architectures?
- How can two models performing the same task differ in latency and confidence?

TransformerLab became a way for me to experiment with all of these ideas in one place.

---

# Features

## 1. Sentiment Analysis

Classifies text according to its sentiment using a fine-tuned DistilBERT model.

This helped me understand how a pretrained Transformer can be fine-tuned for sequence classification.

**Model:** DistilBERT SST-2

---

## 2. Zero-Shot Classification

Allows text to be classified into labels provided by the user without training a new classifier specifically for those categories.

For example:

```text
Text:
Apple announced a new processor for its laptops.

Labels:
technology, sports, finance, healthcare
```

The model ranks the candidate labels according to their relevance.

This was particularly interesting because it demonstrated how Natural Language Inference can be reused for classification.

**Model:** BART Large MNLI

---

## 3. Named Entity Recognition

Identifies entities such as:

- People
- Organizations
- Locations
- Miscellaneous named entities

Example:

```text
Sundar Pichai works at Google in California.
```

The model identifies the entities and returns their type and confidence.

This helped me explore token-level classification instead of only classifying an entire sentence.

**Model:** BERT-based NER

---

## 4. Extractive Question Answering

Given a context and a question, the model finds the answer directly inside the supplied context.

Example:

```text
Context:
Transformers were introduced in the paper
"Attention Is All You Need" in 2017.

Question:
When were Transformers introduced?

Answer:
2017
```

This helped me understand how extractive QA differs from generative question answering.

The model is not generating an arbitrary answer. It predicts the relevant span inside the provided context.

**Model:** DistilBERT fine-tuned on SQuAD

---

# Generative NLP

## 5. Text Summarization

The summarization module converts longer passages into shorter versions.

I experimented with different summary lengths:

- Short
- Medium
- Detailed

The interface also shows information such as original word count, summary word count, and compression.

This helped me understand sequence-to-sequence Transformer architectures.

**Model:** DistilBART CNN

---

## 6. Machine Translation

The translation module uses pretrained MarianMT models.

Supported experiments include:

- English → French
- English → German
- English → Hindi
- French → English
- German → English

Working with translation helped me understand how encoder-decoder Transformers process an input sequence and generate a different output sequence.

**Models:** Helsinki-NLP MarianMT models

---

## 7. Text Generation

The generation module allows a prompt to be continued by an autoregressive language model.

Users can experiment with:

- Maximum generated tokens
- Temperature
- Top-p sampling
- Number of generated sequences

This was useful for understanding how generation parameters affect model output.

For example, changing temperature can make generation more deterministic or more varied.

**Model:** SmolLM2

---

# Semantic Search

One of the main things I wanted to understand was how modern semantic search works.

Traditional keyword search mainly depends on matching words.

Semantic search instead represents text as numerical vectors called **embeddings**.

TransformerLab follows this basic pipeline:

```text
Document
   ↓
Text Chunking
   ↓
Sentence Transformer
   ↓
Chunk Embeddings

User Query
   ↓
Sentence Transformer
   ↓
Query Embedding

Query Embedding
        ↓
Cosine Similarity
        ↓
Ranked Document Chunks
```

The document is divided into smaller chunks.

Each chunk is converted into an embedding using a Sentence Transformer.

The query is also converted into an embedding.

Cosine similarity is then used to compare the query vector with the document vectors.

The most semantically similar chunks are returned.

**Embedding Model:** all-MiniLM-L6-v2

Building this module helped me understand the basic retrieval idea used in many modern AI search and RAG systems.

---

# Document Intelligence

After experimenting with semantic search on plain text, I extended the same idea to documents.

TransformerLab supports:

```text
PDF
TXT
```

The document workflow is:

```text
PDF / TXT
     ↓
Text Extraction
     ↓
Text Cleaning
     ↓
Page-Aware Chunking
     ↓
Sentence Embeddings
     ↓
Cosine Similarity
     ↓
Top Relevant Passages
```

For PDFs, page information is preserved so retrieved chunks can also point back to their page.

This part helped me understand why document processing is an important step before retrieval.

I also learned that document AI has practical limitations. For example, a scanned PDF may contain images of text instead of extractable text. OCR would be required for those documents, which is intentionally outside the current version of this project.

---

# Model Lab

I did not want the project to only call models and display predictions.

I also wanted to inspect some of the things happening underneath.

So I added a small **Model Lab**.

## Tokenizer Inspector

The tokenizer inspector shows how text is converted into tokens.

For example:

```text
Transformer models are interesting.
```

may be transformed into a sequence of tokens and token IDs before being passed into the neural network.

The interface displays:

- Tokens
- Token IDs
- Number of tokens
- Vocabulary size
- Tokenizer type

This helped me understand that Transformer models do not directly process normal words or sentences. They operate on numerical token representations.

---

## Architecture Information

The Model Lab also exposes available model configuration information such as:

- Architecture type
- Hidden dimension
- Transformer layers
- Attention heads
- Parameter count
- Trainable parameters
- Approximate model memory

This made it easier to compare the scale and structure of different Transformer models.

---

# Model Comparison

I also wanted to see what happens when two Transformer models perform the same task.

The project compares:

```text
DistilBERT
vs
BERT
```

for sentiment analysis.

The comparison includes:

| Metric | Purpose |
|---|---|
| Prediction | Output produced by each model |
| Confidence | Confidence of the prediction |
| Latency | Approximate inference time |
| Parameters | Number of model parameters |
| Memory | Approximate model memory |

This experiment helped me understand one of the main ideas behind model compression.

A larger model is not automatically the best choice for every application.

Smaller models can sometimes provide useful performance while requiring fewer computational resources.

The latency measurements are runtime-dependent and should not be treated as formal benchmarks.

Similarly, model confidence is not the same thing as overall model accuracy.

---

# Transformer Architectures I Explored

Through this project I worked with three important categories of Transformer architecture.

### Encoder-Only Transformers

Examples:

```text
BERT
DistilBERT
```

Useful for language understanding tasks such as:

- Classification
- Named Entity Recognition
- Extractive Question Answering

---

### Encoder-Decoder Transformers

Examples:

```text
BART
MarianMT
```

Useful when one sequence needs to be transformed into another sequence.

Examples include:

- Summarization
- Translation

---

### Decoder-Only Transformers

Example:

```text
SmolLM2
```

Useful for autoregressive text generation.

The model predicts new tokens based on the tokens that came before them.

---

# Models Used

| Task | Model / Family |
|---|---|
| Sentiment Analysis | DistilBERT SST-2 |
| Zero-Shot Classification | BART Large MNLI |
| Named Entity Recognition | BERT NER |
| Question Answering | DistilBERT SQuAD |
| Summarization | DistilBART CNN |
| Translation | Helsinki-NLP MarianMT |
| Text Generation | SmolLM2 |
| Semantic Search | all-MiniLM-L6-v2 |
| Model Comparison | DistilBERT + BERT SST-2 |

---

# Tech Stack

### AI / NLP

- PyTorch
- Hugging Face Transformers
- Sentence Transformers

### NLP Models

- BERT
- DistilBERT
- BART
- MarianMT
- SmolLM2
- MiniLM

### Retrieval

- Sentence Embeddings
- Cosine Similarity
- scikit-learn
- NumPy

### Document Processing

- PyPDF

### Data

- Pandas

### Interface

- Gradio

---

# Project Structure

```text
TransformerLab-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── nlp_tasks.py
│   ├── generation.py
│   ├── semantic_search.py
│   ├── document_processor.py
│   └── model_insights.py
│
├── notebooks/
│   └── TransformerLab_AI.ipynb
│
├── examples/
│   └── sample_document.txt
│
└── assets/
    ├── architecture/
    └── screenshots/
```

The notebook contains the experimentation/development work, while the `src` directory contains the modular implementation used by the final application.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/aryaa-pawar/TransformerLab-AI.git
cd TransformerLab-AI
```

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start TransformerLab

```bash
python app.py
```

Gradio will display the local address for the application.

The first execution of a task may take longer because its pretrained model may need to be downloaded.

---

# What I Learned

The biggest takeaway from this project was that **"Transformers" are not just one model or one use case**.

The same overall idea behind the Transformer architecture can be adapted to very different problems.

Through the project I got hands-on experience with:

- Hugging Face pipelines
- Loading pretrained models
- Transformer inference
- Tokenization
- Token IDs
- Encoder-only models
- Decoder-only models
- Encoder-decoder models
- Sequence classification
- Token classification
- Extractive QA
- Natural Language Inference
- Sequence-to-sequence generation
- Autoregressive generation
- Generation parameters
- Sentence embeddings
- Vector representations of text
- Cosine similarity
- Semantic retrieval
- Document chunking
- Overlapping chunks
- PDF text extraction
- Page-aware retrieval
- Parameter counting
- Model memory estimation
- Inference latency
- Comparing Transformer models
- Lazy model loading
- Building a modular ML project instead of keeping everything inside a notebook
- Connecting multiple ML experiments through a Gradio interface

More importantly, building each module separately helped me understand **why different models are selected for different NLP problems**, rather than simply calling one general-purpose AI API for everything.

---

# What This Project Is — and Isn't

TransformerLab AI is primarily an **exploration and learning project**.

It was built to try different Transformer models, understand their use cases, and bring those experiments together into one interactive application.

It is **not intended to be a production NLP platform**.

There are several things I intentionally kept simple:

- Models are pretrained rather than trained from scratch.
- Semantic search uses in-memory embeddings rather than a vector database.
- PDF processing supports text-based PDFs rather than full OCR.
- Latency results are experimental and hardware-dependent.
- Model comparison is for exploration rather than formal benchmarking.
- The project runs locally through Gradio.

Keeping the system relatively simple made it easier to focus on understanding the underlying NLP and Transformer concepts.

---

# Possible Future Experiments

There are several directions I may explore later:

- OCR for scanned documents
- Additional language models
- More embedding models
- Vector databases
- Retrieval-Augmented Generation
- Attention visualization
- Embedding visualization
- Quantized models
- GPU vs CPU benchmarking
- Additional model comparison experiments

These are possible extensions rather than requirements for the current project.

---

# Final Note

This project started simply because I wanted to **try out Transformers properly instead of only reading about them**.

What began as separate experiments with sentiment analysis, NER, question answering, summarization and generation gradually became a single workbench for exploring different parts of modern NLP.

TransformerLab AI represents that learning process: experimenting with pretrained models, understanding their architectures, looking at how text becomes tokens and embeddings, comparing models, and finally connecting everything into one application.
