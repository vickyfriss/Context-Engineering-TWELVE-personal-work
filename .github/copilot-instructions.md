# Copilot Instructions for `twelve-gpt-educational`

## What this repository does

This project is a Streamlit-based educational framework for **data-driven chatbots**.
It demonstrates a retrieval-augmented workflow where the app:

1. loads and transforms structured data,
2. visualizes distributions and selected entities,
3. synthesizes descriptive text from metrics,
4. answers follow-up user questions with LLM chat + embedding retrieval.

It includes multiple demo apps:

- **Football Scout** (player analytics)
- **World Value Survey** (country-level social factor analysis)
- **Personality Test** (person-level profile analysis)
- **Embedding Tool** (builds parquet embedding files from describe datasets)

## Primary architecture

- `pages/*.py` are Streamlit page entrypoints.
- `classes/data_source.py` loads/cleans datasets and computes statistics (z-scores, ranks).
- `classes/data_point.py` holds domain entities (player/country/person).
- `classes/visual.py` builds charts for selected entities vs distributions.
- `classes/description.py` constructs prompts and synthesizes summaries.
- `classes/chat.py` handles conversational state + LLM response generation.
- `classes/embeddings.py` loads/searches embedded Q/A support data.
- `utils/page_components.py` defines shared Streamlit page config/sidebar/CSS.
- `utils/utils.py` has selectors/helpers used across pages.

## LLM and embeddings behavior

- Provider selection is controlled by `settings.py` using Streamlit secrets:
	- `USE_GEMINI = true` uses Google Gemini (`google-generativeai`).
	- otherwise uses Azure OpenAI (`openai==0.28.1` API style).
- Chat and description flows support both providers.
- Embedding search is cosine similarity over stored parquet embeddings under `data/embeddings/`.

## Data and prompt assets

- Raw and prepared datasets are in `data/`.
- Prompt/example/description assets are in:
	- `data/describe/`
	- `data/gpt_examples/`
	- `data/embeddings/` (generated)
- Model cards are in `model cards/`.
- Evaluation scripts for offline analysis live in `evaluation/`.

## How to run locally

Typical app run command:

```bash
streamlit run app.py
```

Required secrets are read from `.streamlit/secrets.toml` (not committed).

## Coding conventions for this repo

- Keep changes **minimal and localized**; avoid broad refactors.
- Preserve current style and dependency choices unless asked otherwise.
- Do not remove dual-provider support (Gemini + Azure OpenAI).
- For Streamlit pages, respect rerun semantics and `st.session_state` chat flow.
- Prefer existing helper patterns (`add_common_page_elements`, `create_chat`, selectors).
- Keep data file paths and naming conventions stable (`data/describe` -> `data/embeddings` mapping).
- This repository is for teaching context-engineering concepts, not coding best-practice hardening.
- Do **not** introduce fallback logic unless the user explicitly asks for it.
- Do **not** add extra typing checks, defensive type conversions, or coercion layers unless the user explicitly asks for them.
- Do **not** run compile/build steps unless the user explicitly asks.
- Do **not** run any git command unless the user explicitly asks.
- Do **not** build small helper functions that are only used once, unless the user explicitly asks for them.
- If any code execution is needed, activate the local virtual environment first
## Common safe change patterns

- New chatbot page:
	1. add a new `pages/<name>.py` entrypoint,
	2. reuse `add_common_page_elements()`,
	3. build/load a `Data`/`Stats` source,
	4. construct `Description`, `Visual`, and `Chat` flow,
	5. add a link in `utils/page_components.py`.
- Updating retrieval quality:
	- edit source Q/A in `data/describe/*`,
	- regenerate embeddings via `pages/embedder.py`.

## Pitfalls to avoid

- Do not hardcode secrets or API keys.
- Do not assume only one LLM provider.
- Do not break existing column names expected by descriptions/visuals.
- Avoid changing `chat_state_hash` logic unless explicitly required.
- `evaluation/analysis_pipeline.py` contains machine-specific paths; treat as offline tooling, not app runtime.

## Testing and validation guidance

- For UI/logic changes, prefer targeted smoke validation by running Streamlit page flows.
- For embedding changes, verify parquet output exists and similarity search still returns results.
- Keep generated files out of unrelated commits unless intentionally updated.
