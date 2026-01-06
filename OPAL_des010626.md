<specification>
## <app_description>

### 1. Overview

This application is an interactive, agent-driven “FDA 510(k) Agentic Review WOW Studio”. It is designed to help regulatory professionals, data analysts, and product teams review FDA 510(k) submissions and related medical device data using multiple large language models (LLMs) and configurable agents.

The original implementation is a Streamlit app with four main functional areas:

1. **Agent Runner** – executes configurable review agents defined in `agents.yaml` against uploaded or pasted submission materials.
2. **Dashboard** – visualizes agent run history (runs, tokens, models).
3. **FDA Data Lab** – ingests, previews, and analyzes FDA-related datasets (GUDID, 510(k), recalls, tracking, etc.) with LLM assistance and basic visualizations.
4. **AI Note Keeper** – a workspace for transforming, summarizing, translating, expanding, and extracting action items from notes, driven by LLMs.

The app supports:

- Multiple LLM providers (OpenAI, Google Gemini, Anthropic Claude, xAI Grok) with model selection and per-provider API key management.
- Two UI languages: English (`en`) and Traditional Chinese (`zh-tw`).
- Visual themes modeled as “painter styles” (e.g., Van Gogh, Monet), controlling accent colors and background gradients.
- Session-based state management for user settings, documents, datasets, run logs, and chat history.

In an OPAL Google (or similar AI application builder) context, the goal is to recreate this behavior as a web-based, agentic application with:

- A **multi-agent execution layer** configured by `agents.yaml`.
- A **skill layer** (documented in `SKILL.md`) that exposes logical operations such as running configured agents, formatting notes, and analyzing FDA datasets.
- A **multi-tab UI** with views corresponding to the original Streamlit tabs.
- Integration with at least one LLM provider (e.g., Gemini) but ideally abstracted so other backends can be added.

The remainder of this specification converts the code into a platform-agnostic description suitable for OPAL, and proposes improved `agents.yaml` and `SKILL.md` that align with the intended behavior.

---

### 2. Core Features & User Experience

#### 2.1 Tabs / Main Screens

The app has four primary screens (tabs):

1. **Agent Runner**
   - Upload or paste submission documents (PDFs converted to text externally, text, markdown).
   - Choose a specific agent from a catalog defined in `agents.yaml`.
   - Provide or reuse input (e.g., previous agent output or merged documents).
   - Optionally override the model, max tokens, and temperature.
   - Run the agent and view its structured markdown output; edit and save that output as input to the next agent.
   - Maintain a **run log** including timestamp, agent, model, tokens (when available), and duration.

2. **Dashboard**
   - Show aggregated statistics from the run log:
     - Total number of runs.
     - Number of unique agents used.
     - Sum of reported tokens.
   - Provide a tabular view of the raw run log.
   - Plot bar chart: tokens used by agent.
   - Plot time series: runs over time per agent and model.

3. **FDA Data Lab**
   - Ingest FDA-related datasets via:
     - File upload (CSV, JSON, TXT).
     - Pasted content (CSV, JSON, plaintext).
   - Tag datasets with a **dataset type** such as GUDID, 510(k) clearance, recalls, tracking, etc.
   - For tabular data:
     - Show overall shape (rows, columns).
     - Show filtered preview (with simple keyword filter applied to text columns).
     - Compute and display descriptive metrics.
     - Build simple visualizations: numeric distributions and categorical top-N counts.
   - For non-tabular data:
     - Provide a text-only preview.
   - Build a **combined textual context** summarizing all loaded datasets (column types, missingness, sample rows).
   - Let the user:
     - Choose an analysis model.
     - Customize an FDA analysis prompt.
     - Decide whether to continue conversation history or start fresh.
   - Use an LLM to generate a comprehensive analysis over all datasets.
   - Maintain a **chat-style history** of analyses, with the ability to clear history and view the full transcript.

4. **AI Note Keeper**
   - Paste raw text or markdown notes.
   - Transform into a simple markdown bullet list baseline.
   - Switch between markdown and raw text editing modes.
   - Apply “AI Magics”:
     - Formatting into well-structured markdown.
     - Keyword highlighting.
     - Summarization.
     - Bidirectional EN ↔ Traditional Chinese translation.
     - Expansion / elaboration.
     - Extraction of actionable tasks.
   - Each magic uses a chosen LLM model and produces markdown/text stored in the current note.

#### 2.2 Global UI Elements

- **Sidebar**
  - Language selection: English / Traditional Chinese.
  - Theme selection: Light / Dark.
  - Painter style selection: choose or randomize a style which adjusts accent color and background gradient.
  - API key management:
    - Show one row per provider (OpenAI, Gemini, Anthropic, Grok).
    - Prefer environment key if available; allow user-supplied keys.
    - Display a status badge (ON/OFF) depending on whether an effective key exists.
  - Global LLM settings:
    - Global model from a fixed list:
      - `gpt-4o-mini`
      - `gpt-4.1-mini`
      - `gemini-2.5-flash`
      - `gemini-2.5-flash-lite`
      - `gemini-3-flash-preview`
      - `claude-3-5-sonnet-20241022`
      - `claude-3-5-haiku-20241022`
      - `grok-4-fast-reasoning`
      - `grok-3-mini`
    - Global max tokens (default 12000).
    - Global temperature (default 0.2).

- **Status Header**
  - API connectivity status: ON if any provider has an effective API key.
  - Document status: whether any submission materials have been uploaded/pasted.
  - Agent run count: number of runs so far.

#### 2.3 Internationalization (i18n)

The application uses a simple key-based translation dictionary. For each UI text key, translations exist for:

- `en` – English.
- `zh-tw` – Traditional Chinese.

The selected language from the sidebar drives text rendering across tabs.

---

### 3. LLM & Agent Architecture

#### 3.1 LLM Provider Abstraction

The original code uses an `LLMProviderManager` to:

- Select a provider based on model name:
  - `gpt-*` → OpenAI.
  - `gemini-*` → Google Generative AI.
  - `claude*` / names containing `anthropic` → Anthropic.
  - names containing `grok` → xAI Grok.
- Resolve an effective API key:
  - If the user provided a key in session state for that provider, use it.
  - Else, fall back to the environment variable for that provider.
- Perform a **chat completion** call:
  - Accepts `model`, `messages`, `max_tokens`, and `temperature`.
  - Returns `content`, `tokens_used`, `duration`, `provider`.

In the OPAL implementation, this should be represented as an abstract **LLM chat skill** that can route to the appropriate backend based on configuration. At minimum, support a primary provider (e.g., Gemini) with a clear path to extend to the others.

#### 3.2 Agent Execution

Agents are configured in `agents.yaml` and executed through an `AgentExecutor` abstraction:

- `load_agents_config()` loads `agents.yaml` and returns a structure:

  ```yaml
  defaults:
    max_tokens: 12000
    temperature: 0.2
  agents:
    <agent_id>:
      name: ...
      skill_number: ...
      category: ...
      description: ...
      default_model: ...
      system_prompt: ...
  ```

- `AgentExecutor`:
  - Provides `list_agents()` that returns all agents sorted by `skill_number`.
  - Provides `execute(agent_id, user_input, model_override, max_tokens, temperature)`:
    - Pulls the agent configuration.
    - Builds a 2-message chat:
      - system: agent’s `system_prompt`.
      - user: `user_input` or a default instruction if empty.
    - Sends to `LLMProviderManager.chat_completion`.
    - Returns `status`, `output`, `model`, `tokens_used`, `duration_seconds`.

In OPAL, this corresponds to a **run_preconfigured_agent** skill (see `SKILL.md` below), which takes an `agent_id`, merges it with the current context (e.g., submission documents, prior outputs), and calls the underlying LLM.

---

### 4. State Management & Data Models

In the Streamlit version, state is stored in `st.session_state`. In OPAL, similar per-user session state should exist. Key logical state elements:

- **User Preferences**
  - `lang` – `"en"` or `"zh-tw"`.
  - `theme` – `"light"` or `"dark"`.
  - `painter_style` – one of the defined painter styles.
  - `api_keys` – mapping from provider name to user-supplied API keys.
  - `global_model`, `global_max_tokens`, `global_temperature`.

- **Agent Layer**
  - `agents_config` – loaded from `agents.yaml`.
  - `run_log` – list of run entries:
    - `timestamp` (UTC).
    - `agent_id`.
    - `model`.
    - `tokens_used`.
    - `duration`.
    - `status`.

- **Documents & Agent Output**
  - `uploaded_docs_raw` – list of:
    - `name`.
    - `size`.
    - `content` (plain text).
  - `last_agent_output` – latest agent output, editable and reusable as input.

- **Notes**
  - `note_content` – current note’s markdown or text content.
  - `note_view_mode` – `"markdown"` or `"text"`.

- **FDA Data Lab**
  - `fda_datasets` – list of dataset entries:
    - `name`.
    - `category` – one of enumerated dataset types.
    - `df` or equivalent structured representation (if tabular).
    - `raw` – original text representation.
    - `source_type` – `"csv"`, `"json"`, `"text"`, etc.
  - `fda_chat_history` – list of messages: `{role: 'user'|'assistant', content: string}`.
  - `fda_prompt` – the customizable analysis prompt.

For OPAL, these can be modeled as persistent JSON-serializable structures in the user session. Skills should accept and/or update these structures as needed (e.g., `run_agent`, `analyze_fda_datasets`, `transform_note`).

---

### 5. Improved `agents.yaml`

Below is a proposed **improved `agents.yaml`** that is consistent with the code and tailored to FDA 510(k) workflows. It expands the set of agents, clarifies descriptions, and standardizes prompts.

All entries marked with `Assumed:` or `Inferred:` are design choices made from the context of the code.

<agents_yaml>

```yaml
# agents.yaml
# Assumed: This file defines the configurable review agents used by the Agent Runner tab.
# The application loads this at startup and exposes each agent in the UI.

defaults:
  # Global default parameters for all agents.
  max_tokens: 8000          # Assumed: Enough for detailed 510(k) analyses.
  temperature: 0.2          # Low temperature for more deterministic, regulatory-style output.
  doc_truncation_chars: 8000  # Inferred: Soft limit for concatenated document input.

agents:
  intake_triage:
    name: "Submission Intake & Triage"
    skill_number: 1
    category: "Preliminary Screening"
    default_model: "gemini-2.5-flash"     # Assumed: Fast, cost-effective default.
    description: >
      Quickly screens a 510(k) submission to identify basic completeness,
      device type, intended use, and obvious gaps in documentation.
      Produces a structured overview and triage recommendations.
    system_prompt: |
      You are an experienced FDA 510(k) intake reviewer.

      TASK:
      - Analyze the provided submission materials.
      - Identify the device type, intended use, indications for use, and key components.
      - Highlight which core elements of a 510(k) file appear present or missing (e.g., device description, 
        predicate device information, substantial equivalence discussion, labeling, performance testing).
      - Flag any obvious administrative or content gaps that would block further review.
      - Suggest priority areas for deeper follow-up agents.

      OUTPUT FORMAT (Markdown):
      - "## Device Overview"
      - "## Intended Use & Indications"
      - "## Completeness Checklist" (use a table with columns: Section, Present?, Comments)
      - "## Immediate Concerns / Red Flags"
      - "## Recommended Follow-Up Actions"

      Be concise but specific. Do not invent data that is not supported by the text.

  predicate_device_mapper:
    name: "Predicate Device Mapper"
    skill_number: 2
    category: "Substantial Equivalence"
    default_model: "gpt-4o-mini"
    description: >
      Extracts and structures information about claimed predicate devices,
      including product codes, manufacturer, and comparison points.
    system_prompt: |
      You are an FDA reviewer focusing on predicate devices and substantial equivalence.

      TASK:
      - From the submission materials, identify all predicate devices referenced.
      - For each predicate, extract:
        - Device name
        - Manufacturer (if available)
        - 510(k) number (if available)
        - Product code(s)
        - Classification regulation number and class (if stated)
      - Summarize how the subject device is claimed to be similar to or different from each predicate.
      - Call out any ambiguities or missing information that could impact the substantial equivalence argument.

      OUTPUT FORMAT (Markdown):
      - "## Predicate Device List" (table with columns: Identifier, Manufacturer, 510(k) Number, Product Code, Class/Reg)
      - "## Comparison Summary"
      - "## Gaps / Questions for Sponsor"

      If predicate information is not provided, state this clearly.

  substantial_equivalence_reviewer:
    name: "Substantial Equivalence Reviewer"
    skill_number: 3
    category: "Substantial Equivalence"
    default_model: "claude-3-5-sonnet-20241022"
    description: >
      Performs a structured review of the substantial equivalence rationale,
      including technological characteristics, indications, and performance testing.
    system_prompt: |
      You are an expert FDA 510(k) reviewer assessing substantial equivalence (SE).

      TASK:
      - Evaluate the SE discussion in the submission:
        - Indications for use comparison.
        - Technological characteristics comparison.
        - Performance / bench / clinical testing summary.
      - Identify where the subject device is the same as or different from the predicate(s).
      - Assess whether provided data appears adequate to support SE, and where additional information may be needed.

      OUTPUT FORMAT (Markdown):
      - "## Indications for Use Comparison"
      - "## Technological Characteristics Comparison"
      - "## Performance / Testing Summary"
      - "## Preliminary SE Assessment"
      - "## Information Gaps and Questions"

      Do not make regulatory decisions; instead, provide a reasoned assessment and list clear questions.

  labeling_and_ifu_checker:
    name: "Labeling & IFU Checker"
    skill_number: 4
    category: "Labeling"
    default_model: "gemini-3-flash-preview"
    description: >
      Reviews labeling and instructions for use (IFU) for clarity, consistency with intended use,
      and presence of appropriate warnings, precautions, and contraindications.
    system_prompt: |
      You are an FDA labeling reviewer for medical devices.

      TASK:
      - Focus on labeling, instructions for use (IFU), and any patient/provider-facing materials.
      - Check for:
        - Consistency with the stated intended use and indications.
        - Presence of key warnings, precautions, and contraindications for this device type.
        - Clarity of instructions: step-by-step usability, critical steps, and risk-relevant instructions.
      - Identify any misleading or promotional statements that go beyond the cleared intended use (if inferable).
      - Suggest specific labeling improvements.

      OUTPUT FORMAT (Markdown):
      - "## Labeling Overview"
      - "## Key Warnings / Precautions / Contraindications"
      - "## Usability & Clarity Assessment"
      - "## Potential Misleading Statements"
      - "## Recommended Labeling Edits"

      If labeling excerpts are limited, clearly note what is missing.

  risk_benefit_profile_summarizer:
    name: "Risk–Benefit Profile Summarizer"
    skill_number: 5
    category: "Risk Management"
    default_model: "grok-4-fast-reasoning"
    description: >
      Synthesizes the risk–benefit profile of the device based on adverse events, recalls,
      and safety-related sections, potentially using FDA datasets loaded in the Data Lab.
    system_prompt: |
      You are an FDA risk analyst evaluating the overall risk–benefit profile of a medical device.

      TASK:
      - Based on the submission content (and, if available, any summarized FDA datasets about recalls
        or adverse events), construct a narrative risk–benefit profile.
      - Identify:
        - Foreseeable hazards and hazardous situations.
        - Key mitigations (design, labeling, training, etc.).
        - Any patterns in recalls, adverse events, or safety communications (if described in the input).
      - Comment on whether the described mitigations appear aligned with identified risks.

      OUTPUT FORMAT (Markdown):
      - "## Intended Clinical Benefits"
      - "## Identified Risks and Hazards"
      - "## Mitigations and Controls"
      - "## Observed Safety Signals (if any)"
      - "## Overall Risk–Benefit Narrative"

      Do not invent specific event counts or rates that are not in the input; instead, characterize qualitatively.

  executive_summary_writer:
    name: "Executive Summary Writer"
    skill_number: 6
    category: "Summary"
    default_model: "gpt-4.1-mini"
    description: >
      Produces a concise, executive-level summary of the 510(k) submission,
      suitable for briefing decision makers or capturing key review findings.
    system_prompt: |
      You are preparing an executive-level FDA 510(k) review summary.

      TASK:
      - Summarize the main elements of the submission:
        - Device description and intended use.
        - Predicate devices and high-level SE rationale.
        - Key performance / testing results (without excessive technical detail).
        - Major risks and mitigations.
        - Notable labeling considerations.
      - Incorporate, where possible, insights produced by prior agents (intake, SE review, labeling, risk profile)
        when they are included in the input.

      OUTPUT FORMAT (Markdown):
      - "## Device & Intended Use"
      - "## Predicate Devices & SE Rationale"
      - "## Performance Evidence Summary"
      - "## Risk & Safety Considerations"
      - "## Labeling Highlights"
      - "## Open Issues / Follow-Up Items"

      Aim for 1–2 pages of well-structured markdown suitable for senior review.
```

</agents_yaml>

---

### 6. Improved `SKILL.md`

Below is a proposed **improved `SKILL.md`** that abstracts the core operations of the application into reusable skills. These skills should be implemented in OPAL’s backend and exposed to the agent system.

<skill_md>

```markdown
# SKILL.md – FDA 510(k) Agentic Review WOW Studio

This skill catalog describes the main capabilities required to recreate the FDA 510(k) Agentic Review WOW Studio in OPAL.

All JSON schemas below are illustrative; OPAL builders should adapt them to the platform’s preferred type system.

---

## 1. `llm_chat`

**Purpose**

Unified interface to send a chat-style prompt to an LLM, abstracting over provider, model, and basic parameters.

**Input**

```json
{
  "model": "string",
  "messages": [
    { "role": "system" | "user" | "assistant", "content": "string" }
  ],
  "max_tokens": 0,
  "temperature": 0.0
}
```

- `model`: Logical model name, e.g. `gemini-2.5-flash`.
- `messages`: Conversation history in standard chat format.
- `max_tokens`: Upper bound on new tokens (assumed: default 8000 if omitted).
- `temperature`: Sampling temperature (assumed default 0.2).

**Output**

```json
{
  "content": "string",
  "tokens_used": 0,
  "provider": "string",
  "duration_seconds": 0.0
}
```

**Behavior**

- Route to the correct provider based on `model`.
- Use appropriate API keys (user-supplied or environment).
- Return the assistant’s full reply and basic usage metadata.

---

## 2. `run_agent`

**Purpose**

Execute a configured review agent from `agents.yaml` with supplied input text and optional overrides.

**Input**

```json
{
  "agent_id": "string",
  "input_text": "string",
  "model_override": "string | null",
  "max_tokens": 0,
  "temperature": 0.0
}
```

**Output**

```json
{
  "status": "success" | "error",
  "agent_id": "string",
  "model": "string",
  "output": "string",
  "tokens_used": 0,
  "duration_seconds": 0.0,
  "error_message": "string | null"
}
```

**Behavior**

1. Look up the agent by `agent_id` in `agents.yaml`.
2. Choose the model:
   - `model_override` if provided,
   - else `agent.default_model`,
   - else global default.
3. Construct messages:
   - `system`: agent’s `system_prompt`.
   - `user`: `input_text` or a default “use configured behavior” instruction if blank.
4. Call `llm_chat`.
5. Return structured result (status, output, metrics).

**Error Handling**

- If `agent_id` is unknown, return `status = "error"` with an appropriate message.
- Surface API or network errors in `error_message`.

---

## 3. `append_run_log_entry`

**Purpose**

Record metadata for each agent execution to power the dashboard.

**Input**

```json
{
  "timestamp_utc": "string",         // ISO 8601
  "agent_id": "string",
  "model": "string",
  "tokens_used": 0,
  "duration_seconds": 0.0,
  "status": "success" | "error"
}
```

**Output**

```json
{ "ok": true }
```

**Behavior**

Append the entry to the per-user run log store. This log is later used for metrics and visualizations.

---

## 4. `ingest_submission_documents`

**Purpose**

Store submission-related documents (uploaded or pasted) as raw text for use by agents.

**Input**

```json
{
  "documents": [
    {
      "name": "string",
      "size_bytes": 0,
      "content": "string"
    }
  ]
}
```

**Output**

```json
{
  "stored_count": 0,
  "total_documents": 0
}
```

**Behavior**

- Normalize encoding to UTF-8.
- Append documents to the user’s `uploaded_docs` collection.
- Return how many were stored and the new total count.

---

## 5. `get_submission_document_context`

**Purpose**

Build a concatenated text context out of stored submission documents, truncated to a configured maximum length.

**Input**

```json
{
  "max_chars": 8000
}
```

**Output**

```json
{
  "context": "string"
}
```

**Behavior**

- Concatenate all document `content` fields separated by blank lines.
- If the result exceeds `max_chars`, truncate and append a marker such as `"[TRUNCATED DOCUMENT CONTENT]"`.
- Used as default input for agents if the user does not provide a custom input.

---

## 6. `transform_note`

**Purpose**

Apply an LLM-powered transformation (“magic”) to the current note.

**Supported operations**

- `"format"` – clean, structure, and format as markdown.
- `"keywords"` – highlight given keywords using HTML span tags.
- `"summary"` – produce a concise summary.
- `"translate"` – auto-detect EN vs. Traditional Chinese and translate in the opposite direction.
- `"expand"` – expand and elaborate the content.
- `"actions"` – extract and list action items.

**Input**

```json
{
  "operation": "format" | "keywords" | "summary" | "translate" | "expand" | "actions",
  "note_content": "string",
  "model": "string",
  "keywords": ["string"],
  "highlight_color": "string"
}
```

**Output**

```json
{
  "transformed_content": "string"
}
```

**Behavior**

1. Construct a specific system prompt depending on `operation` mirroring the original code logic.
2. Prepare messages:
   - `system`: operation-specific guidelines.
   - `user`: `note_content` plus any keyword context.
3. Call `llm_chat`.
4. Return the resulting markdown or text.

**Error Handling**

- If `operation` is unsupported, return the original `note_content` unchanged (or an explicit error, depending on platform preference).

---

## 7. `ingest_fda_datasets`

**Purpose**

Store one or more FDA-related datasets for subsequent exploration and analysis.

**Input**

```json
{
  "datasets": [
    {
      "name": "string",
      "category": "GUDID" | "Medical device classification" | "510(k) clearance" | "Medical device tracking (breast implant)" | "Guidance" | "Medical device recall" | "510(k) review notes" | "Safety notice" | "Other",
      "source_type": "csv" | "json" | "text",
      "raw": "string",
      "parsed_table": {
        "columns": ["string"],
        "rows": [
          ["string"]
        ]
      }
    }
  ]
}
```

- `parsed_table` is optional; if omitted or parsing fails, treat dataset as text-only.

**Output**

```json
{
  "stored_count": 0,
  "total_datasets": 0
}
```

**Behavior**

- Persist datasets to the user’s FDA dataset collection.
- Allow subsequent selection and inspection in the UI.

---

## 8. `build_fda_dataset_context`

**Purpose**

Generate a textual description of all stored FDA datasets (columns, types, missingness, and sample rows) for use in LLM analysis.

**Input**

```json
{
  "max_chars": 8000
}
```

**Output**

```json
{
  "context": "string"
}
```

**Behavior**

For each dataset:

- Output header with name, category, and source type.
- If tabular:
  - Provide row/column counts.
  - For up to N columns (e.g., 40), report:
    - Column name.
    - Inferred dtype (string, numeric, date, etc.).
    - Approximate missing value ratio.
  - Include a small CSV-like sample (e.g., 10 rows).
- If text-only:
  - Include a truncated text preview.
- Concatenate all dataset summaries separated by delimiters.
- Truncate combined context to `max_chars` with a clear truncation marker.

---

## 9. `analyze_fda_datasets`

**Purpose**

Run an LLM analysis over the current FDA datasets using a customizable analysis prompt and optional conversation history.

**Input**

```json
{
  "analysis_model": "string",
  "user_prompt": "string",
  "dataset_context": "string",
  "max_tokens": 0,
  "temperature": 0.2,
  "prior_messages": [
    { "role": "user" | "assistant", "content": "string" }
  ]
}
```

**Output**

```json
{
  "assistant_analysis": "string",
  "tokens_used": 0,
  "duration_seconds": 0.0
}
```

**Behavior**

1. Construct messages:
   - `system`: fixed role description as an FDA officer and data analyst, emphasizing non-hallucination of columns.
   - Append `prior_messages` if conversation continuation is desired.
   - `user`: concatenation of `user_prompt` and `dataset_context`.
2. Call `llm_chat`.
3. Return the assistant’s analysis and usage metadata.

**Conversation Management**

- A higher-level workflow should store the new user and assistant messages into `fda_chat_history`.

---

## 10. `reset_fda_chat_history`

**Purpose**

Clear the conversation history for FDA dataset analyses.

**Input**

```json
{}
```

**Output**

```json
{ "ok": true }
```

---

## 11. `compute_dashboard_metrics`

**Purpose**

Generate aggregate statistics from the agent run log for display in the dashboard.

**Input**

```json
{
  "run_log": [
    {
      "timestamp_utc": "string",
      "agent_id": "string",
      "model": "string",
      "tokens_used": 0
    }
  ]
}
```

**Output**

```json
{
  "total_runs": 0,
  "unique_agents": 0,
  "total_tokens": 0,
  "tokens_by_agent": [
    { "agent_id": "string", "tokens_used": 0 }
  ],
  "runs_over_time": [
    { "timestamp_utc": "string", "agent_id": "string", "model": "string" }
  ]
}
```

**Behavior**

- Compute counts and aggregates analogous to the original tables and charts.
- Serve as the data source for UI visualizations.

---

## 12. `set_user_preferences`

**Purpose**

Update and persist user preferences such as language, theme, painter style, and global LLM parameters.

**Input**

```json
{
  "lang": "en" | "zh-tw",
  "theme": "light" | "dark",
  "painter_style": "string",
  "global_model": "string",
  "global_max_tokens": 0,
  "global_temperature": 0.0
}
```

**Output**

```json
{ "ok": true }
```

**Behavior**

- Merge the provided preferences into the user’s session and/or profile.
- Ensure subsequent skill calls use the updated global defaults where applicable.
```

</skill_md>

---

### 7. Application Flow Examples

#### 7.1 Common Agent Workflow

1. User uploads several submission documents (or pastes text).
2. OPAL calls `ingest_submission_documents`.
3. User opens **Agent Runner**, selects the **Submission Intake & Triage** agent.
4. OPAL calls `get_submission_document_context` to build a default input.
5. User optionally edits this input in the UI.
6. When the user clicks “Run agent”:
   - OPAL calls `run_agent` with:
     - `agent_id = "intake_triage"`.
     - `input_text` from the text area.
     - `model_override`, `max_tokens`, `temperature` from UI or defaults.
   - On success, OPAL calls `append_run_log_entry` to store metadata.
7. UI shows the agent’s markdown output and gives the user the option to edit it and save for the next agent run.
8. User then selects another agent (e.g., **Executive Summary Writer**) and uses the previous output as new input, continuing the multi-agent review chain.

#### 7.2 FDA Data Lab Workflow

1. User uploads several CSV files tagged as “510(k) clearance” and “Medical device recall”.
2. OPAL calls `ingest_fda_datasets` with parsed tabular data where possible.
3. User selects a dataset to inspect; the UI renders:
   - Row/column metrics.
   - Filtered preview.
   - Basic distribution and top-N categorical charts (driven by `parsed_table`).
4. User configures a custom prompt for cross-dataset risk signal analysis and chooses an analysis model.
5. OPAL calls `build_fda_dataset_context` to produce a textual summary of all datasets.
6. OPAL calls `analyze_fda_datasets` with:
   - `analysis_model`.
   - `user_prompt`.
   - `dataset_context`.
   - `prior_messages` if conversation continuation is enabled.
7. The returned `assistant_analysis` is displayed; a conversation transcript is built from `prior_messages` plus the new turn.
8. If the user clicks “Clear analysis conversation”, OPAL calls `reset_fda_chat_history`.

#### 7.3 AI Note Keeper Workflow

1. User pastes rough meeting notes or review comments into the Note Keeper.
2. The UI converts them into a simple bullet list baseline (can be implemented client-side or with `transform_note` and `operation = "format"`).
3. User chooses view mode (“Markdown” or “Text”) and edits as desired.
4. User invokes:
   - `transform_note` with `operation = "summary"` to create a concise summary, or
   - `transform_note` with `operation = "actions"` to extract action items.
5. Updated content overwrites the current note state; the preview is rerendered.

---

### 8. Security & Configuration Considerations

- **API Keys**
  - Store per-provider keys securely, never return them to the client.
  - Allow using environment-scoped keys as defaults with optional user override.
  - For OPAL, use secure secret management for provider credentials.

- **Data Privacy**
  - All uploaded documents and datasets must be scoped to the authenticated user or workspace.
  - Do not share user documents or generated analyses across users unless explicitly configured for collaboration.

- **LLM Usage**
  - Include system prompts that explicitly instruct the model to avoid hallucinating data or columns not present in the input.
  - Consider logging LLM usage only in aggregate for monitoring; avoid logging sensitive raw content unless necessary and consented.

- **Rate Limiting & Error Handling**
  - Implement per-user rate limiting for expensive operations like `run_agent` and `analyze_fda_datasets`.
  - Provide user-friendly error messages when a provider is unavailable, a model name is invalid, or tokens are exhausted.

---

### 9. Missing Information & Assumptions

- **Original `agents.yaml` and `SKILL.md`** were not provided; the versions in this specification are **inferred** and designed for best practice FDA 510(k) workflows.
- **Authentication & Multi-user Handling** are not defined in the source code; OPAL should assume a per-user or per-workspace identity layer.
- **File Types & Parsing**:
  - Original code expects PDFs but does not show PDF text extraction; OPAL should either:
    - Integrate a PDF-to-text service, or
    - Restrict uploads to pre-converted text/markdown.
- **FDA-Specific Taxonomies** (product codes, regulations, etc.) are not encoded; agents rely on LLM world knowledge and the provided datasets rather than a curated ontology.

These assumptions can be refined based on organization-specific requirements.

---

### 10. Follow-Up Questions (for Requirements Refinement)

1. Should the OPAL version of the app support **all four** LLM providers (OpenAI, Gemini, Anthropic, Grok) from day one, or is it acceptable to start with Gemini only and leave others as future extensions?
2. What is the **expected scale and sensitivity** of submission documents and FDA datasets (e.g., internal drafts vs. publicly available data), and are there any regulatory constraints on where data and logs may be stored?
3. Do you want **multi-user collaboration features** (shared datasets, shared run logs, shared notes), or is the initial scope strictly per-user and private?
4. Should the agents operate purely as **single-step LLM prompts**, or do you envision multi-step reasoning workflows (e.g., agent chains or tools) that require additional orchestration beyond what’s described here?
5. Are there specific **FDA guidance documents, templates, or checklists** that agents should explicitly reference or structure outputs around (e.g., specific 510(k) guidance by device category)?
6. How important is **versioning and auditability** of agent configurations (`agents.yaml`) and system prompts (e.g., storing which version produced a given analysis)?
7. Should the **run log and dashboard** be persisted across sessions (e.g., in a database) or can they be ephemeral, cleared when a user session ends?
8. Do you need **fine-grained access control** on datasets (e.g., certain users may see only anonymized or aggregated views), especially when dealing with potentially sensitive recall or adverse event data?
9. Should the **AI Note Keeper** support multiple named notes/documents with a list view, or is a single note workspace sufficient for your initial use case?
10. Are there **performance constraints** or limits on dataset size (rows, columns, file size) that OPAL should enforce for the FDA Data Lab to keep interactions responsive?
11. Would you like the app to support **exporting outputs** (e.g., markdown reports or dashboards) to external systems such as SharePoint, Google Drive, or a document management system?
12. Should agents and skills be able to **call each other programmatically** (e.g., the Executive Summary agent automatically invoking prior agents) or should orchestration remain fully manual via the UI?
13. Do you require **fine-tuned or domain-adapted models** for specific device categories, or is reliance on general-purpose LLMs (with good prompts) acceptable?
14. Are there any **specific FDA datasets or APIs** (beyond file uploads) that you want to integrate directly (e.g., openFDA, GUDID web services), and should those be modeled as additional skills?
15. How strict must the application be about **non-hallucination**? Do you want mechanisms such as explicit data citations, traceability to source rows, or confidence scores attached to LLM outputs?
16. Is support for **additional languages** beyond English and Traditional Chinese required in the medium term (e.g., Simplified Chinese, Japanese, EU languages)?
17. Would you like an **admin UI** for managing `agents.yaml` (creating, editing, disabling agents) from within the app, rather than editing YAML files externally?
18. Are there specific **compliance frameworks** (e.g., 21 CFR Part 11, ISO 13485, ISO 14971) that the system needs to align with in terms of logging, electronic records, and user authentication?
19. How important is **theme and visual customization** (painter styles, dark mode) relative to core functionality—should OPAL prioritize exact replication of the WOW Studio look, or is a simpler design acceptable?
20. Do you foresee needing **API-level access** to the app’s capabilities (e.g., programmatic access to run agents and analyze datasets) so that other systems can integrate with this 510(k) review workflow?

</app_description>
</specification>
