import os
import time
import random
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml

# Optional imports – wrapped in try/except in LLM manager
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import anthropic
except Exception:
    anthropic = None

import httpx


# ------------- GLOBAL CONSTANTS ------------------------------------------------

APP_TITLE = "FDA 510(k) Agentic Review WOW Studio"

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

PAINTER_STYLES = {
    "Van Gogh": {"accent": "#FFB300", "bg": "linear-gradient(135deg,#1e3c72,#2a5298)"},
    "Monet": {"accent": "#5EC2F2", "bg": "linear-gradient(135deg,#b2fefa,#0ed2f7)"},
    "Da Vinci": {"accent": "#CDAA7D", "bg": "linear-gradient(135deg,#2c3e50,#bdc3c7)"},
    "Picasso": {"accent": "#FF4B81", "bg": "linear-gradient(135deg,#000000,#434343)"},
    "Matisse": {"accent": "#1ABC9C", "bg": "linear-gradient(135deg,#1abc9c,#16a085)"},
    "Klimt": {"accent": "#F1C40F", "bg": "linear-gradient(135deg,#f1c40f,#e67e22)"},
    "Hokusai": {"accent": "#2980B9", "bg": "linear-gradient(135deg,#2c3e50,#2980b9)"},
    "Frida Kahlo": {"accent": "#E74C3C", "bg": "linear-gradient(135deg,#e74c3c,#8e44ad)"},
    "Rembrandt": {"accent": "#A67C52", "bg": "linear-gradient(135deg,#3c2a21,#8e5a2a)"},
    "Dali": {"accent": "#F39C12", "bg": "linear-gradient(135deg,#f39c12,#d35400)"},
    "Cézanne": {"accent": "#3498DB", "bg": "linear-gradient(135deg,#3498db,#2ecc71)"},
    "Renoir": {"accent": "#E67E22", "bg": "linear-gradient(135deg,#f39c12,#e67e22)"},
    "Turner": {"accent": "#F5B041", "bg": "linear-gradient(135deg,#f5b041,#f7dc6f)"},
    "Goya": {"accent": "#7F8C8D", "bg": "linear-gradient(135deg,#2c3e50,#7f8c8d)"},
    "Basquiat": {"accent": "#F1C40F", "bg": "linear-gradient(135deg,#000000,#f1c40f)"},
    "Pollock": {"accent": "#E74C3C", "bg": "linear-gradient(135deg,#2c3e50,#e74c3c)"},
    "O'Keeffe": {"accent": "#9B59B6", "bg": "linear-gradient(135deg,#9b59b6,#e91e63)"},
    "Chagall": {"accent": "#8E44AD", "bg": "linear-gradient(135deg,#0f2027,#8e44ad)"},
    "Vermeer": {"accent": "#2980B9", "bg": "linear-gradient(135deg,#f1c40f,#2980b9)"},
    "Michelangelo": {"accent": "#D35400", "bg": "linear-gradient(135deg,#2c3e50,#d35400)"},
}

FDA_DATASET_TYPES = [
    "GUDID",
    "Medical device classification",
    "510(k) clearance",
    "Medical device tracking (breast implant)",
    "Guidance",
    "Medical device recall",
    "510(k) review notes",
    "Safety notice",
    "Other",
]


# ------------- INTERNATIONALIZATION -------------------------------------------

def get_i18n_dict() -> Dict[str, Dict[str, str]]:
    return {
        "app_title": {
            "en": "FDA 510(k) Agentic Review WOW Studio",
            "zh-tw": "FDA 510(k) 智慧審查 WOW 工作室",
        },
        "tab_agents": {"en": "Agent Runner", "zh-tw": "代理人執行器"},
        "tab_dashboard": {"en": "Dashboard", "zh-tw": "儀表板"},
        "tab_notes": {"en": "AI Note Keeper", "zh-tw": "AI 筆記管家"},
        "tab_fda_data": {"en": "FDA Data Lab", "zh-tw": "FDA 數據實驗室"},
        "sidebar_language": {"en": "Language", "zh-tw": "語言"},
        "sidebar_theme": {"en": "Theme", "zh-tw": "主題"},
        "sidebar_style": {"en": "Painter Style", "zh-tw": "畫風樣式"},
        "sidebar_jackpot": {"en": "Jackpot Style", "zh-tw": "隨機大獎樣式"},
        "sidebar_api_keys": {"en": "API Keys", "zh-tw": "API 金鑰"},
        "sidebar_llm_settings": {"en": "LLM Settings", "zh-tw": "LLM 設定"},
        "light": {"en": "Light", "zh-tw": "亮色"},
        "dark": {"en": "Dark", "zh-tw": "暗色"},
        "provider_openai": {"en": "OpenAI", "zh-tw": "OpenAI"},
        "provider_gemini": {"en": "Gemini", "zh-tw": "Gemini"},
        "provider_anthropic": {"en": "Anthropic", "zh-tw": "Anthropic"},
        "provider_grok": {"en": "Grok", "zh-tw": "Grok"},
        "api_from_env": {"en": "Using environment key", "zh-tw": "使用環境變數金鑰"},
        "api_enter": {"en": "Enter API key", "zh-tw": "請輸入 API 金鑰"},
        "model": {"en": "Model", "zh-tw": "模型"},
        "max_tokens": {"en": "Max tokens", "zh-tw": "最大 tokens"},
        "temperature": {"en": "Temperature", "zh-tw": "溫度"},
        "agent_select": {"en": "Select agent", "zh-tw": "選擇代理人"},
        "agent_input": {"en": "Agent input (you can edit previous output here)", "zh-tw": "代理人輸入（可在此編輯上一個輸出）"},
        "run_agent": {"en": "Run agent", "zh-tw": "執行代理人"},
        "use_last_output": {"en": "Use last agent output as input", "zh-tw": "使用上一個代理人輸出作為輸入"},
        "save_for_next": {"en": "Save as input for next agent", "zh-tw": "儲存為下一個代理人輸入"},
        "output_markdown": {"en": "Markdown view", "zh-tw": "Markdown 檢視"},
        "output_text": {"en": "Text view (editable)", "zh-tw": "文字檢視（可編輯）"},
        "status_section": {"en": "WOW Status Indicators", "zh-tw": "WOW 狀態指標"},
        "status_api": {"en": "API Connectivity", "zh-tw": "API 連線狀態"},
        "status_docs": {"en": "Documents", "zh-tw": "文件"},
        "status_runs": {"en": "Agent Runs", "zh-tw": "代理人執行次數"},
        "notes_paste": {"en": "Paste your text / markdown", "zh-tw": "貼上文字或 Markdown"},
        "notes_transform": {"en": "Transform to organized markdown", "zh-tw": "轉換為結構化 Markdown"},
        "notes_model": {"en": "Model for note operations", "zh-tw": "筆記處理模型"},
        "notes_view_mode": {"en": "Note view mode", "zh-tw": "筆記檢視模式"},
        "view_markdown": {"en": "Markdown", "zh-tw": "Markdown"},
        "view_text": {"en": "Text", "zh-tw": "文字"},
        "ai_magics": {"en": "AI Magics", "zh-tw": "AI 魔法"},
        "magic_format": {"en": "AI Formatting", "zh-tw": "AI 格式優化"},
        "magic_keywords": {"en": "AI Keywords Highlighter", "zh-tw": "AI 關鍵字標示"},
        "magic_summary": {"en": "AI Summary", "zh-tw": "AI 摘要"},
        "magic_translate": {"en": "AI EN ↔ 繁中 Translate", "zh-tw": "AI 中英互譯"},
        "magic_expand": {"en": "AI Expansion / Elaboration", "zh-tw": "AI 擴寫說明"},
        "magic_actions": {"en": "AI Action Items", "zh-tw": "AI 行動清單"},
        "keywords_input": {"en": "Keywords (comma separated)", "zh-tw": "關鍵字（以逗號分隔）"},
        "keyword_color": {"en": "Keyword color", "zh-tw": "關鍵字顏色"},
        # FDA Data Lab specific
        "fda_intro": {
            "en": "Upload/paste FDA medical device datasets (GUDID, 510(k), recalls, tracking, etc.) for automatic summary and interactive analysis.",
            "zh-tw": "上傳或貼上 FDA 醫療器材資料集（GUDID、510(k)、召回、追蹤等），自動產生摘要與互動分析。",
        },
        "fda_upload_title": {"en": "Upload FDA datasets", "zh-tw": "上傳 FDA 資料集"},
        "fda_dataset_type": {"en": "Dataset type", "zh-tw": "資料集類型"},
        "fda_paste_title": {"en": "Paste dataset", "zh-tw": "貼上資料集"},
        "fda_paste_format": {"en": "Pasted data format", "zh-tw": "貼上資料格式"},
        "fda_select_dataset": {"en": "Select dataset to inspect", "zh-tw": "選擇要檢視的資料集"},
        "fda_no_datasets": {"en": "No FDA datasets loaded yet.", "zh-tw": "尚未載入任何 FDA 資料集。"},
        "fda_overview": {"en": "Dataset overview", "zh-tw": "資料集總覽"},
        "fda_analysis_model": {"en": "Model for FDA data analysis", "zh-tw": "FDA 數據分析模型"},
        "fda_analysis_prompt": {"en": "Custom analysis prompt", "zh-tw": "自訂分析提示詞"},
        "fda_run_summary": {"en": "Run / refine FDA analysis", "zh-tw": "執行／優化 FDA 分析"},
        "fda_continue_chat": {"en": "Continue from previous analysis (keep conversation history)", "zh-tw": "延續先前分析（保留對話紀錄）"},
        "fda_clear_chat": {"en": "Clear analysis conversation", "zh-tw": "清除分析對話"},
        "fda_visual_config": {"en": "Interactive visualization", "zh-tw": "互動視覺化"},
        "fda_filter": {"en": "Row filter keyword (applied to text columns)", "zh-tw": "列篩選關鍵字（套用於文字欄位）"},
    }


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return get_i18n_dict().get(key, {}).get(lang, key)


# ------------- SESSION STATE INIT & THEME -------------------------------------

def init_session_state():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    if "painter_style" not in st.session_state:
        st.session_state["painter_style"] = "Van Gogh"
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}
    if "agents_config" not in st.session_state:
        st.session_state["agents_config"] = load_agents_config()
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []
    if "last_agent_output" not in st.session_state:
        st.session_state["last_agent_output"] = ""
    if "note_content" not in st.session_state:
        st.session_state["note_content"] = ""
    if "note_view_mode" not in st.session_state:
        st.session_state["note_view_mode"] = "markdown"
    # FDA Data Lab state
    if "fda_datasets" not in st.session_state:
        st.session_state["fda_datasets"] = []  # list of dicts: name, category, df, raw, source_type
    if "fda_chat_history" not in st.session_state:
        st.session_state["fda_chat_history"] = []  # list of OpenAI-style messages
    if "fda_prompt" not in st.session_state:
        st.session_state["fda_prompt"] = default_fda_prompt()


def apply_theme():
    theme = st.session_state.get("theme", "light")
    style_name = st.session_state.get("painter_style", "Van Gogh")
    style_cfg = PAINTER_STYLES.get(style_name, PAINTER_STYLES["Van Gogh"])
    accent = style_cfg["accent"]
    bg = style_cfg["bg"]

    base_bg = "#111111" if theme == "dark" else "#FFFFFF"
    base_text = "#F5F5F5" if theme == "dark" else "#111111"

    css = f"""
    <style>
      .stApp {{
        background: {bg};
        background-attachment: fixed;
      }}
      .main-wrapping-container {{
        background: {base_bg}CC;
        padding: 1.5rem;
        border-radius: 1rem;
      }}
      .wow-badge {{
        display:inline-block;
        padding:0.15rem 0.45rem;
        border-radius:999px;
        font-size:0.7rem;
        margin-right:0.15rem;
        background:{accent}22;
        color:{accent};
        border:1px solid {accent}66;
      }}
      .wow-chip-ok {{
        background:#2ecc7122;
        color:#2ecc71;
        border:1px solid #2ecc7166;
      }}
      .wow-chip-warn {{
        background:#f1c40f22;
        color:#f1c40f;
        border:1px solid #f1c40f66;
      }}
      .wow-chip-bad {{
        background:#e74c3c22;
        color:#e74c3c;
        border:1px solid #e74c3c66;
      }}
      h1, h2, h3, h4, h5, h6, p, span, label {{
        color:{base_text};
      }}
      .coral-keyword {{
        color: coral;
        font-weight: 600;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------- LLM PROVIDER MANAGER ------------------------------------------

class LLMProviderManager:
    def __init__(self):
        self.env_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "grok": os.getenv("GROK_API_KEY"),
        }

    def get_effective_key(self, provider: str) -> Optional[str]:
        user_key = st.session_state["api_keys"].get(provider)
        if user_key:
            return user_key
        return self.env_keys.get(provider)

    def identify_provider(self, model: str) -> str:
        if model.startswith("gpt-"):
            return "openai"
        if model.startswith("gemini-"):
            return "gemini"
        if "claude" in model or "anthropic" in model:
            return "anthropic"
        if "grok" in model:
            return "grok"
        return "openai"

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 12000,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        provider = self.identify_provider(model)
        api_key = self.get_effective_key(provider)
        if not api_key:
            raise RuntimeError(f"No API key found for provider: {provider}")

        start = time.time()
        content = ""
        tokens_used = None

        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not available")
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            tokens_used = getattr(resp.usage, "total_tokens", None)

        elif provider == "gemini":
            if genai is None:
                raise RuntimeError("google-generativeai package not available")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            # join all messages into single prompt with simple roles
            joined = []
            for m in messages:
                prefix = "System: " if m["role"] == "system" else "User: "
                joined.append(prefix + m["content"])
            prompt = "\n".join(joined)
            resp = model_obj.generate_content(prompt)
            content = resp.text

        elif provider == "anthropic":
            if anthropic is None:
                raise RuntimeError("anthropic package not available")
            client = anthropic.Anthropic(api_key=api_key)
            sys_prompt = ""
            user_content = ""
            for m in messages:
                if m["role"] == "system":
                    sys_prompt += m["content"] + "\n"
                else:
                    user_content += m["content"] + "\n"
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_prompt.strip(),
                messages=[{"role": "user", "content": user_content.strip()}],
            )
            content = "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"])
            tokens_used = getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0)

        elif provider == "grok":
            # Use OpenAI-compatible HTTP endpoint for xAI Grok
            headers = {"Authorization": f"Bearer {api_key}"}
            json = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            with httpx.Client(timeout=120) as client:
                r = client.post("https://api.x.ai/v1/chat/completions", headers=headers, json=json)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens")

        duration = time.time() - start
        return {"content": content, "tokens_used": tokens_used, "duration": duration, "provider": provider}


# ------------- AGENT EXECUTOR -------------------------------------------------

def load_agents_config() -> Dict[str, Any]:
    path = Path("agents.yaml")
    if not path.exists():
        return {"agents": {}, "defaults": {"max_tokens": 12000, "temperature": 0.2}}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AgentExecutor:
    def __init__(self, llm_manager: LLMProviderManager, agents_config: Dict[str, Any]):
        self.llm_manager = llm_manager
        self.config = agents_config or {"agents": {}, "defaults": {}}
        self.defaults = self.config.get("defaults", {})

    def list_agents(self) -> List[Dict[str, Any]]:
        agents = []
        for agent_id, cfg in self.config.get("agents", {}).items():
            agents.append({"id": agent_id, **cfg})
        # sort by skill_number if present
        agents.sort(key=lambda a: a.get("skill_number", 999))
        return agents

    def execute(
        self,
        agent_id: str,
        user_input: str,
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        agents = self.config.get("agents", {})
        if agent_id not in agents:
            raise RuntimeError(f"Unknown agent: {agent_id}")
        cfg = agents[agent_id]
        system_prompt = cfg.get("system_prompt", "")
        model = model_override or cfg.get("default_model") or st.session_state.get("global_model", MODEL_OPTIONS[0])
        max_tokens = max_tokens or self.defaults.get("max_tokens", 12000)
        temperature = temperature if temperature is not None else self.defaults.get("temperature", 0.2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input.strip() or "Use your configured behavior with the current context."},
        ]

        resp = self.llm_manager.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "status": "success",
            "output": resp["content"],
            "model": model,
            "tokens_used": resp.get("tokens_used"),
            "duration_seconds": resp.get("duration"),
        }


# ------------- SIDEBAR (THEME, LANGUAGE, API, MODEL) -------------------------

def render_sidebar(llm_manager: LLMProviderManager):
    with st.sidebar:
        st.markdown("### 🎨 WOW Studio")
        lang = st.radio(
            t("sidebar_language"),
            ["en", "zh-tw"],
            index=0 if st.session_state["lang"] == "en" else 1,
            horizontal=True,
        )
        st.session_state["lang"] = lang

        theme = st.radio(
            t("sidebar_theme"),
            [t("light"), t("dark")],
            index=0 if st.session_state["theme"] == "light" else 1,
            horizontal=True,
        )
        st.session_state["theme"] = "light" if theme == t("light") else "dark"

        style = st.selectbox(
            t("sidebar_style"),
            list(PAINTER_STYLES.keys()),
            index=list(PAINTER_STYLES.keys()).index(st.session_state["painter_style"]),
        )
        st.session_state["painter_style"] = style
        if st.button("🎰 " + t("sidebar_jackpot")):
            st.session_state["painter_style"] = random.choice(list(PAINTER_STYLES.keys()))
            st.experimental_rerun()

        st.markdown("---")
        st.markdown(f"### 🔐 {t('sidebar_api_keys')}")

        def key_row(provider_key: str, label_key: str, env_var: str):
            env_val = llm_manager.env_keys.get(provider_key)
            st.caption(f"{t(label_key)} ({env_var})")
            col1, col2 = st.columns([3, 2])
            with col1:
                if env_val:
                    # env key exists; don't show the key
                    st.success("✅ " + t("api_from_env"))
                else:
                    api_val = st.text_input(
                        t("api_enter"),
                        type="password",
                        key=f"api_{provider_key}",
                    )
                    if api_val:
                        st.session_state["api_keys"][provider_key] = api_val
            with col2:
                effective = llm_manager.get_effective_key(provider_key)
                status_class = "wow-chip-ok" if effective else "wow-chip-bad"
                status_label = "ON" if effective else "OFF"
                st.markdown(
                    f'<span class="wow-badge {status_class}">{status_label}</span>',
                    unsafe_allow_html=True,
                )

        key_row("openai", "provider_openai", "OPENAI_API_KEY")
        key_row("gemini", "provider_gemini", "GEMINI_API_KEY")
        key_row("anthropic", "provider_anthropic", "ANTHROPIC_API_KEY")
        key_row("grok", "provider_grok", "GROK_API_KEY")

        st.markdown("---")
        st.markdown(f"### 🤖 {t('sidebar_llm_settings')}")
        model = st.selectbox(t("model"), MODEL_OPTIONS, index=0)
        st.session_state["global_model"] = model
        max_tokens = st.number_input(t("max_tokens"), min_value=256, max_value=120000, value=12000, step=256)
        st.session_state["global_max_tokens"] = int(max_tokens)
        temperature = st.slider(t("temperature"), min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        st.session_state["global_temperature"] = float(temperature)


# ------------- WOW STATUS INDICATORS & DASHBOARD -----------------------------

def render_status_header(llm_manager: LLMProviderManager):
    st.markdown("#### ⚡ " + t("status_section"))
    col1, col2, col3 = st.columns(3)

    with col1:
        any_api = any(llm_manager.get_effective_key(p) for p in ["openai", "gemini", "anthropic", "grok"])
        cls = "wow-chip-ok" if any_api else "wow-chip-bad"
        label = "OK" if any_api else "Missing"
        st.markdown(
            f'{t("status_api")}<br><span class="wow-badge {cls}">{label}</span>',
            unsafe_allow_html=True,
        )
    with col2:
        docs_loaded = bool(st.session_state.get("uploaded_docs_raw"))
        cls = "wow-chip-ok" if docs_loaded else "wow-chip-warn"
        label = "Loaded" if docs_loaded else "None"
        st.markdown(
            f'{t("status_docs")}<br><span class="wow-badge {cls}">{label}</span>',
            unsafe_allow_html=True,
        )
    with col3:
        runs = len(st.session_state["run_log"])
        cls = "wow-chip-ok" if runs > 0 else "wow-chip-warn"
        st.markdown(
            f'{t("status_runs")}<br><span class="wow-badge {cls}">{runs} run(s)</span>',
            unsafe_allow_html=True,
        )


def render_dashboard_tab():
    import pandas as pd
    import altair as alt

    st.markdown("### 📊 Interactive Review Dashboard")

    logs = st.session_state.get("run_log", [])
    if not logs:
        st.info("No agent runs yet. Execute an agent run first.")
        return

    df = pd.DataFrame(logs)
    with st.expander("Raw run log"):
        st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total runs", len(df))
    with col2:
        st.metric("Unique agents", df["agent_id"].nunique())
    with col3:
        st.metric("Total tokens (if reported)", int(df["tokens_used"].fillna(0).sum()))

    # Tokens by agent
    if "tokens_used" in df.columns:
        chart_data = (
            df.groupby("agent_id")["tokens_used"]
            .sum()
            .reset_index()
            .sort_values("tokens_used", ascending=False)
        )
        st.markdown("#### Tokens by agent")
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("agent_id", sort="-y", title="Agent"),
                y=alt.Y("tokens_used", title="Tokens"),
                tooltip=["agent_id", "tokens_used"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # Runs over time
    if "timestamp" in df.columns:
        st.markdown("#### Runs over time")
        chart2 = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x="timestamp:T",
                y="count()",
                tooltip=["timestamp", "agent_id", "model"],
                color="agent_id",
            )
            .properties(height=300)
        )
        st.altair_chart(chart2, use_container_width=True)


# ------------- AGENT RUNNER TAB ----------------------------------------------

def render_agent_runner_tab(executor: AgentExecutor):
    import pandas as pd  # needed for timestamp formatting in run_log

    st.markdown("### 🧠 " + t("tab_agents"))

    # Document upload / paste
    if "uploaded_docs_raw" not in st.session_state:
        st.session_state["uploaded_docs_raw"] = []

    with st.expander("📁 Upload or paste submission materials", expanded=False):
        files = st.file_uploader(
            "Upload PDFs / text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if files:
            for f in files:
                try:
                    content = f.read()
                    try:
                        text = content.decode("utf-8", errors="ignore")
                    except AttributeError:
                        text = content
                    st.session_state["uploaded_docs_raw"].append(
                        {"name": f.name, "size": f.size, "content": text}
                    )
                    st.success(f"Loaded {f.name}")
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")

        paste = st.text_area("Paste text", height=150)
        if st.button("Save pasted as document"):
            if paste.strip():
                st.session_state["uploaded_docs_raw"].append(
                    {
                        "name": f"Pasted-{len(st.session_state['uploaded_docs_raw'])+1}",
                        "size": len(paste),
                        "content": paste,
                    }
                )
                st.success("Pasted content saved.")
            else:
                st.warning("Nothing to save.")

        docs = st.session_state["uploaded_docs_raw"]
        if docs:
            st.markdown("**Loaded documents:**")
            for d in docs[-5:]:
                st.markdown(f"- {d['name']} ({len(d['content'])} chars)")

    # Agent selection
    agents = executor.list_agents()
    if not agents:
        st.error("No agents found in agents.yaml")
        return

    agent_labels = [
        f"[#{a.get('skill_number','?')}] {a.get('name','')} ({a['id']})" for a in agents
    ]
    idx = st.selectbox(
        t("agent_select"),
        list(range(len(agents))),
        format_func=lambda i: agent_labels[i],
    )
    agent_cfg = agents[idx]

    with st.expander("Agent details", expanded=False):
        st.markdown(f"**ID:** `{agent_cfg['id']}`")
        st.markdown(f"**Skill #:** {agent_cfg.get('skill_number','?')}")
        st.markdown(f"**Category:** {agent_cfg.get('category','')}")
        st.markdown(f"**Description:** {agent_cfg.get('description','')}")
        st.markdown("**System prompt (truncated):**")
        sys_prompt = agent_cfg.get("system_prompt", "")
        st.code(textwrap.shorten(sys_prompt, width=1200, placeholder="..."))

    # Input source
    col_a, col_b = st.columns([2, 1])
    with col_b:
        use_last = st.checkbox(t("use_last_output"), value=False)
    default_input = ""
    if use_last and st.session_state["last_agent_output"]:
        default_input = st.session_state["last_agent_output"]
    else:
        # Default: concat docs (trimmed)
        docs = st.session_state.get("uploaded_docs_raw", [])
        merged = "\n\n".join(d["content"] for d in docs)
        default_input = textwrap.shorten(
            merged,
            width=8000,
            placeholder="\n\n[TRUNCATED DOCUMENT CONTENT]\n",
        )

    with col_a:
        user_input = st.text_area(
            t("agent_input"),
            value=default_input,
            height=220,
        )

    # Override model / params
    with st.expander("Override model & parameters (optional)", expanded=False):
        model_override = st.selectbox(
            "Model override",
            ["(use agent default)"] + MODEL_OPTIONS,
            index=0,
            key="agent_model_override",
        )
        if model_override == "(use agent default)":
            model_override = None

        max_tokens_override = st.number_input(
            "Max tokens (blank = global default)",
            min_value=256,
            max_value=120000,
            value=st.session_state.get("global_max_tokens", 12000),
            step=256,
        )
        temp_override = st.slider(
            "Temperature (blank = global default)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("global_temperature", 0.2),
            step=0.05,
        )

    if st.button("🚀 " + t("run_agent"), type="primary"):
        try:
            with st.spinner("Running agent..."):
                res = executor.execute(
                    agent_id=agent_cfg["id"],
                    user_input=user_input,
                    model_override=model_override,
                    max_tokens=int(max_tokens_override),
                    temperature=float(temp_override),
                )
            st.session_state["last_agent_output"] = res["output"]

            # Log
            st.session_state["run_log"].append(
                {
                    "timestamp": pd.Timestamp.utcnow(),
                    "agent_id": agent_cfg["id"],
                    "model": res["model"],
                    "tokens_used": res.get("tokens_used"),
                    "duration": res.get("duration_seconds"),
                    "status": res.get("status"),
                }
            )

            st.success("Agent executed.")
            render_agent_output(res, agent_cfg)
        except Exception as e:
            st.error(f"Error during agent execution: {e}")

    # Always show latest output if exists
    if st.session_state.get("last_agent_output"):
        with st.expander("Latest output", expanded=True):
            render_agent_output(
                {
                    "output": st.session_state["last_agent_output"],
                    "model": "(last)",
                    "tokens_used": None,
                    "duration_seconds": None,
                    "status": "success",
                },
                agent_cfg,
                show_metrics=False,
            )


def render_agent_output(result: Dict[str, Any], agent_cfg: Dict[str, Any], show_metrics: bool = True):
    if show_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", result.get("model", ""))
        with col2:
            st.metric("Tokens", result.get("tokens_used", 0) or 0)
        with col3:
            st.metric("Duration (s)", round(result.get("duration_seconds", 0) or 0, 2))

    tab1, tab2 = st.tabs([t("output_markdown"), t("output_text")])
    with tab1:
        st.markdown(result.get("output", ""), unsafe_allow_html=True)
    with tab2:
        edited = st.text_area(
            t("output_text"),
            value=result.get("output", ""),
            height=260,
        )
        if st.button("💾 " + t("save_for_next")):
            st.session_state["last_agent_output"] = edited
            st.success("Saved for next agent input.")


# ------------- AI NOTE KEEPER TAB --------------------------------------------

def format_note_with_coral_keywords(text: str) -> str:
    # Basic structure; LLM will improve later if user wants.
    lines = [l.strip() for l in text.splitlines()]
    bullets = [f"- {l}" for l in lines if l]
    md = "\n".join(bullets)
    return md


def highlight_keywords(note: str, keywords: List[str], color: str) -> str:
    # naive keyword replacement
    out = note
    for kw in sorted(set(k.strip() for k in keywords if k.strip()), key=len, reverse=True):
        out = out.replace(
            kw,
            f'<span style="color:{color}; font-weight:600;">{kw}</span>',
        )
    return out


def run_note_llm(
    llm_manager: LLMProviderManager,
    note: str,
    magic: str,
    model: str,
    extra: Dict[str, Any] = None,
) -> str:
    extra = extra or {}
    system_prompt = ""
    if magic == "format":
        system_prompt = (
            "You are an expert technical editor. Clean and reformat this note into well-structured markdown with "
            "clear headings, bullet lists, and a short summary. Do not invent new facts."
        )
    elif magic == "keywords":
        kws = extra.get("keywords", [])
        color = extra.get("color", "coral")
        system_prompt = (
            "You are a highlighting engine. Given the note and a list of keywords, return markdown where each exact "
            f"keyword is wrapped in <span style='color:{color}; font-weight:600;'>keyword</span>. Do not change other text."
            f"\n\nKeywords: {', '.join(kws)}"
        )
    elif magic == "summary":
        system_prompt = "Summarize the note into concise bullet points and a short abstract, in markdown."
    elif magic == "translate":
        system_prompt = (
            "Detect if the note is in English or Traditional Chinese and translate it to the OTHER language. "
            "Keep markdown formatting."
        )
    elif magic == "expand":
        system_prompt = (
            "Expand and elaborate the note, adding clarifying explanations and examples, but keep original structure. "
            "Output markdown."
        )
    elif magic == "actions":
        system_prompt = (
            "Extract and list all clear action items from the note. For each action, provide: "
            "1) action statement, 2) owner if mentioned, 3) due date if mentioned, 4) priority if inferable."
            "Output as a markdown task list."
        )
    else:
        return note

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": note},
    ]
    resp = llm_manager.chat_completion(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.1,
    )
    return resp["content"]


def render_note_keeper_tab(llm_manager: LLMProviderManager):
    st.markdown("### 📝 " + t("tab_notes"))

    col1, col2 = st.columns([2, 1])
    with col1:
        raw = st.text_area(t("notes_paste"), height=200)
    with col2:
        if st.button("✨ " + t("notes_transform")):
            if raw.strip():
                st.session_state["note_content"] = format_note_with_coral_keywords(raw)
            else:
                st.warning("Nothing to transform.")

    if not st.session_state["note_content"] and raw.strip():
        # initialize if user typed but didn't click
        st.session_state["note_content"] = format_note_with_coral_keywords(raw)

    st.markdown("---")
    st.markdown("#### 🔧 Note workspace")

    model = st.selectbox(
        t("notes_model"),
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.get("global_model", MODEL_OPTIONS[0])),
        key="notes_model_select",
    )

    view_mode = st.radio(
        t("notes_view_mode"),
        [t("view_markdown"), t("view_text")],
        index=0 if st.session_state["note_view_mode"] == "markdown" else 1,
        horizontal=True,
    )
    st.session_state["note_view_mode"] = "markdown" if view_mode == t("view_markdown") else "text"

    if st.session_state["note_view_mode"] == "markdown":
        edited = st.text_area(
            "Markdown",
            value=st.session_state["note_content"],
            height=260,
        )
        st.session_state["note_content"] = edited
        st.markdown("---")
        st.markdown("Preview:")
        st.markdown(edited, unsafe_allow_html=True)
    else:
        text_view = st.text_area(
            "Text",
            value=st.session_state["note_content"],
            height=260,
        )
        st.session_state["note_content"] = text_view

    st.markdown("---")
    st.markdown("#### 🪄 " + t("ai_magics"))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎯 " + t("magic_format")):
            with st.spinner("Formatting note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "format",
                    model,
                )
        if st.button("🧠 " + t("magic_summary")):
            with st.spinner("Summarizing note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "summary",
                    model,
                )
        if st.button("🌐 " + t("magic_translate")):
            with st.spinner("Translating note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "translate",
                    model,
                )
    with col_b:
        kw_str = st.text_input(t("keywords_input"), "")
        kw_color = st.color_picker(t("keyword_color"), value="#FF7F50")  # coral default
        if st.button("🔎 " + t("magic_keywords")):
            kws = [k.strip() for k in kw_str.split(",") if k.strip()]
            if not kws:
                st.warning("No keywords entered.")
            else:
                with st.spinner("Highlighting keywords..."):
                    st.session_state["note_content"] = run_note_llm(
                        llm_manager,
                        st.session_state["note_content"],
                        "keywords",
                        model,
                        extra={"keywords": kws, "color": kw_color},
                    )
        if st.button("📈 " + t("magic_expand")):
            with st.spinner("Expanding note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "expand",
                    model,
                )
        if st.button("✅ " + t("magic_actions")):
            with st.spinner("Extracting action items..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "actions",
                    model,
                )


# ------------- FDA DATA LAB TAB ----------------------------------------------

def default_fda_prompt() -> str:
    return textwrap.dedent(
        """\
        You are an FDA 510(k) reviewer and medical device data analyst.

        Using ALL currently loaded FDA medical device datasets, produce a comprehensive, structured analysis that covers:

        1. Dataset inventory:
           - What datasets are present (e.g., GUDID, classification, 510(k), recalls, tracking, guidance, safety notices)?
           - For each, summarize size, time coverage, and the most important fields.
        2. Cross-dataset linkages:
           - How can records be joined across datasets (e.g., by device identifiers, product codes, submission numbers)?
           - Highlight any key join paths that are critical for regulatory review.
        3. Risk and safety signals:
           - Identify device types, product codes, or manufacturers with notable recall or safety signal density.
           - Call out patterns involving breast implants or other high-risk devices if visible.
        4. Data quality and completeness:
           - Point out missing data, inconsistent coding, or fields that may limit interpretability.
        5. Recommended next steps:
           - Suggest concrete follow-up analyses or visualizations that would help an FDA officer deepen the review.

        Answer in clear markdown with headings, bullet points, and small tables where helpful. Do not invent columns that are not present in the data; if something is not available, say so explicitly.
        """
    )


def build_fda_dataset_context(datasets: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    parts: List[str] = []
    for idx, ds in enumerate(datasets):
        name = ds.get("name", f"Dataset {idx+1}")
        category = ds.get("category", "Unknown")
        source_type = ds.get("source_type", "unknown")
        header = f"Dataset {idx+1}: [{category}] {name} (source_type={source_type})"
        df = ds.get("df")
        if df is not None:
            try:
                n_rows, n_cols = df.shape
            except Exception:
                n_rows, n_cols = "?", "?"
            col_info_lines = []
            for col in list(df.columns)[:40]:
                series = df[col]
                dtype = str(series.dtype)
                try:
                    missing_ratio = series.isna().mean()
                    missing_str = f"{missing_ratio:.1%}"
                except Exception:
                    missing_str = "n/a"
                col_info_lines.append(f"- {col} (dtype={dtype}, missing={missing_str})")
            try:
                sample_text = df.head(10).to_csv(index=False)
            except Exception:
                sample_text = ""
            part = (
                f"{header}\n"
                f"Rows: {n_rows}, Columns: {n_cols}\n"
                "Columns:\n"
                + "\n".join(col_info_lines)
            )
            if sample_text:
                part += "\nSample rows (CSV):\n" + sample_text
        else:
            raw = ds.get("raw", "")
            sample_text = textwrap.shorten(raw, width=1500, placeholder=" ...[TRUNCATED]")
            part = f"{header}\n(Non-tabular text dataset; preview below)\n\n{sample_text}"
        parts.append(part)

    context = "\n\n---\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n...[TRUNCATED DATASET CONTEXT]..."
    return context


def render_fda_data_lab_tab(llm_manager: LLMProviderManager):
    import pandas as pd
    import altair as alt
    from io import StringIO

    st.markdown("### 🧪 " + t("tab_fda_data"))
    st.markdown(t("fda_intro"))

    # ---------------- Upload & Paste Section ----------------
    with st.expander("📂 " + t("fda_upload_title"), expanded=True):
        col_u1, col_u2 = st.columns(2)

        with col_u1:
            dataset_type = st.selectbox(
                t("fda_dataset_type"),
                FDA_DATASET_TYPES,
                key="fda_dataset_type_select",
            )
            files = st.file_uploader(
                "Upload CSV / JSON / TXT",
                type=["csv", "json", "txt"],
                accept_multiple_files=True,
                key="fda_file_uploader",
            )
            if files:
                for f in files:
                    try:
                        raw_bytes = f.read()
                        text = raw_bytes.decode("utf-8", errors="ignore")
                        ext = f.name.lower().split(".")[-1]
                        df = None
                        source_type = ext
                        if ext == "csv":
                            try:
                                df = pd.read_csv(StringIO(text))
                            except Exception as e:
                                st.warning(f"{f.name}: could not parse CSV ({e}); keeping as text only.")
                        elif ext == "json":
                            try:
                                df = pd.read_json(StringIO(text))
                            except Exception as e:
                                st.warning(f"{f.name}: could not parse JSON ({e}); keeping as text only.")
                        # txt -> no structured parse by default
                        st.session_state["fda_datasets"].append(
                            {
                                "name": f.name,
                                "category": dataset_type,
                                "df": df,
                                "raw": text,
                                "source_type": source_type,
                            }
                        )
                        st.success(f"Loaded dataset: {f.name} ({dataset_type})")
                    except Exception as e:
                        st.error(f"Error reading {f.name}: {e}")

        with col_u2:
            st.markdown("#### " + t("fda_paste_title"))
            paste_type = st.selectbox(
                t("fda_paste_format"),
                ["csv", "json", "text"],
                key="fda_paste_format_select",
            )
            paste_category = st.selectbox(
                t("fda_dataset_type") + " (pasted)",
                FDA_DATASET_TYPES,
                key="fda_paste_dataset_type",
            )
            pasted = st.text_area("Pasted content", height=140, key="fda_paste_area")
            if st.button("➕ " + "Add pasted dataset"):
                if not pasted.strip():
                    st.warning("Nothing to add.")
                else:
                    df = None
                    if paste_type == "csv":
                        try:
                            df = pd.read_csv(StringIO(pasted))
                        except Exception as e:
                            st.warning(f"Could not parse pasted CSV ({e}); keeping as text only.")
                    elif paste_type == "json":
                        try:
                            df = pd.read_json(StringIO(pasted))
                        except Exception as e:
                            st.warning(f"Could not parse pasted JSON ({e}); keeping as text only.")
                    st.session_state["fda_datasets"].append(
                        {
                            "name": f"Pasted-{len(st.session_state['fda_datasets'])+1}",
                            "category": paste_category,
                            "df": df,
                            "raw": pasted,
                            "source_type": paste_type,
                        }
                    )
                    st.success("Pasted dataset added.")

        if st.session_state["fda_datasets"]:
            st.markdown("**Loaded FDA datasets:**")
            for i, ds in enumerate(st.session_state["fda_datasets"][-12:], start=1):
                df = ds.get("df")
                shape_str = ""
                if df is not None:
                    try:
                        r, c = df.shape
                        shape_str = f" | {r}×{c}"
                    except Exception:
                        shape_str = ""
                st.markdown(
                    f"- #{i} [{ds.get('category','Unknown')}] {ds.get('name','(unnamed)')} "
                    f"(source={ds.get('source_type','')}{shape_str})"
                )

    st.markdown("---")

    datasets = st.session_state.get("fda_datasets", [])
    if not datasets:
        st.info(t("fda_no_datasets"))
        return

    # ---------------- Dataset Inspection & Visualization ----------------
    st.markdown("#### " + t("fda_overview"))
    idx = st.selectbox(
        t("fda_select_dataset"),
        options=list(range(len(datasets))),
        format_func=lambda i: f"[{datasets[i].get('category','Unknown')}] {datasets[i].get('name','(unnamed)')}",
        key="fda_selected_dataset_idx",
    )
    selected = datasets[idx]
    df = selected.get("df")

    filter_kw = st.text_input(t("fda_filter"), value="", key="fda_filter_keyword")

    if df is not None:
        df_preview = df.copy()
        if filter_kw.strip():
            try:
                mask = None
                for col in df_preview.select_dtypes(include=["object", "string"]).columns:
                    col_mask = df_preview[col].astype(str).str.contains(filter_kw, case=False, na=False)
                    mask = col_mask if mask is None else (mask | col_mask)
                if mask is not None:
                    df_preview = df_preview[mask]
            except Exception:
                pass

        n_rows, n_cols = df.shape
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", n_rows)
        with col2:
            st.metric("Columns", n_cols)
        with col3:
            st.metric("Filtered rows", len(df_preview))

        st.dataframe(df_preview.iloc[:100, :50], use_container_width=True)

        st.markdown("#### " + t("fda_visual_config"))
        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = [c for c in df.columns if c not in num_cols]

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if num_cols:
                num_col = st.selectbox("Numeric column for distribution", num_cols, key="fda_numeric_col")
                chart = (
                    alt.Chart(df_preview)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=40), title=num_col),
                        y=alt.Y("count():Q", title="Count"),
                        tooltip=[num_col, "count()"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No numeric columns detected in this dataset.")

        with col_v2:
            if cat_cols:
                cat_col = st.selectbox("Categorical column (top categories)", cat_cols, key="fda_cat_col")
                top_n = st.slider("Top N categories", min_value=5, max_value=40, value=15, step=1)
                try:
                    counts = (
                        df_preview[cat_col]
                        .astype(str)
                        .value_counts()
                        .reset_index()
                        .rename(columns={"index": cat_col, cat_col: "count"})
                        .head(top_n)
                    )
                    chart2 = (
                        alt.Chart(counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
                            tooltip=[cat_col, "count"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not build categorical chart: {e}")
            else:
                st.info("No categorical columns detected in this dataset.")
    else:
        st.info("Selected dataset is text-only. It will still be used in the LLM analysis.")
        sample = textwrap.shorten(selected.get("raw", ""), width=1500, placeholder=" ...[TRUNCATED]")
        st.text_area("Preview (read-only)", value=sample, height=200)

    st.markdown("---")

    # ---------------- LLM Analysis & Conversation ----------------
    st.markdown("#### 🧠 FDA dataset analysis (LLM)")

    analysis_model = st.selectbox(
        t("fda_analysis_model"),
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.get("global_model", MODEL_OPTIONS[0]))
        if st.session_state.get("global_model", MODEL_OPTIONS[0]) in MODEL_OPTIONS
        else 0,
        key="fda_analysis_model_select",
    )
    max_tokens = st.number_input(
        t("max_tokens"),
        min_value=512,
        max_value=120000,
        value=st.session_state.get("global_max_tokens", 12000),
        step=512,
        key="fda_analysis_max_tokens",
    )
    continue_chat = st.checkbox(
        t("fda_continue_chat"),
        value=True,
        key="fda_continue_chat_checkbox",
    )

    st.session_state["fda_prompt"] = st.text_area(
        t("fda_analysis_prompt"),
        value=st.session_state.get("fda_prompt", default_fda_prompt()),
        height=220,
        key="fda_analysis_prompt_area",
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        if st.button("🚀 " + t("fda_run_summary"), type="primary", key="fda_run_summary_btn"):
            try:
                with st.spinner("Analyzing FDA datasets with LLM..."):
                    base_system = (
                        "You are an expert FDA medical device officer and data analyst. "
                        "You will receive:\n"
                        "1) a detailed description of multiple FDA-related datasets (GUDID, 510(k), recalls, etc.), and\n"
                        "2) a custom analysis prompt from the user.\n\n"
                        "Use ONLY the provided dataset context; do not hallucinate columns or data that are not present. "
                        "If information is not available, say so clearly. Respond in well-structured markdown."
                    )
                    dataset_context = build_fda_dataset_context(datasets)

                    messages: List[Dict[str, str]] = [{"role": "system", "content": base_system}]
                    if continue_chat and st.session_state["fda_chat_history"]:
                        # preserve prior turns
                        messages.extend(st.session_state["fda_chat_history"])

                    user_content = (
                        st.session_state["fda_prompt"].strip()
                        + "\n\n---\n\nCURRENT DATASET CONTEXT:\n"
                        + dataset_context
                    )
                    messages.append({"role": "user", "content": user_content})

                    resp = llm_manager.chat_completion(
                        model=analysis_model,
                        messages=messages,
                        max_tokens=int(max_tokens),
                        temperature=st.session_state.get("global_temperature", 0.2),
                    )

                # update conversation history
                if continue_chat:
                    st.session_state["fda_chat_history"].append({"role": "user", "content": user_content})
                    st.session_state["fda_chat_history"].append(
                        {"role": "assistant", "content": resp["content"]}
                    )
                else:
                    st.session_state["fda_chat_history"] = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": resp["content"]},
                    ]

                st.markdown("##### Latest analysis")
                st.markdown(resp["content"], unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during FDA data analysis: {e}")

    with col_btn2:
        if st.button("🧹 " + t("fda_clear_chat"), key="fda_clear_chat_btn"):
            st.session_state["fda_chat_history"] = []
            st.success("Cleared FDA analysis conversation history.")

    if st.session_state["fda_chat_history"]:
        with st.expander("📚 Full analysis conversation", expanded=False):
            for msg in st.session_state["fda_chat_history"]:
                role = msg.get("role", "user")
                if role == "user":
                    st.markdown("**User prompt + context:**")
                else:
                    st.markdown("**Assistant analysis:**")
                st.markdown(msg.get("content", ""), unsafe_allow_html=True)
                st.markdown("---")


# ------------- UTIL ----------------------------------------------------------

def pd_timestamp_now():
    import pandas as _pd
    return _pd.Timestamp.utcnow()


# ------------- MAIN ----------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()
    apply_theme()

    llm_manager = LLMProviderManager()
    executor = AgentExecutor(llm_manager=llm_manager, agents_config=st.session_state["agents_config"])

    # Wrap main content for background CSS
    st.markdown('<div class="main-wrapping-container">', unsafe_allow_html=True)

    render_sidebar(llm_manager)
    st.title(t("app_title"))
    render_status_header(llm_manager)

    tabs = st.tabs(
        [
            t("tab_agents"),
            t("tab_dashboard"),
            t("tab_fda_data"),
            t("tab_notes"),
        ]
    )

    with tabs[0]:
        render_agent_runner_tab(executor)
    with tabs[1]:
        render_dashboard_tab()
    with tabs[2]:
        render_fda_data_lab_tab(llm_manager)
    with tabs[3]:
        render_note_keeper_tab(llm_manager)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
