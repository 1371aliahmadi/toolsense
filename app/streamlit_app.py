"""ToolSense — Streamlit UI — Phase 09 (repolished)"""

import re
import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Query type detection (mirrors ranker.py logic, no import needed) ──────────
_BM25_STOPWORDS = {
    "a","an","the","for","and","or","to","of","in","my","from","with",
    "that","this","is","are","as","i","me","can","do","how","what",
    "which","using","use","tool","ai","ml","platform","app","software",
    "help","helper","helps","make","makes","get","your","you","best",
    "free","new","pro","plus","online","powered","based","smart","easy",
}

def _detect_query_type(query: str) -> str:
    words = query.split()
    has_named = any(
        w[0].isupper()
        and w.lower() not in {"ai","the","a","an","i","my","for","to","in"}
        and len(w) > 2
        for w in words[1:]
    )
    if has_named:
        return "named_entity"
    meaningful = [t for t in re.sub(r"[^a-z\s]", "", query.lower()).split()
                  if t not in _BM25_STOPWORDS and len(t) > 1]
    return "short" if len(meaningful) <= 3 else "conversational"

_QUERY_TYPE_META = {
    "named_entity":  ("🏷️ Named entity",   "#fef3c7", "#92400e", "BM25 ↑↑ precision mode"),
    "short":         ("⚡ Short query",     "#eff6ff", "#1e40af", "BM25 ↑ keyword mode"),
    "conversational":("💬 Conversational",  "#f0fdf4", "#065f46", "SBERT ↑ semantic mode"),
}

CATEGORY_EMOJIS = {
    "video": "🎬", "writing": "✍️", "image": "🎨", "code": "💻",
    "developer": "💻", "marketing": "📢", "support": "💬", "chat": "💬",
    "hr": "👥", "recruit": "👥", "language": "🌐", "translat": "🌐",
    "presentation": "📊", "research": "🔬", "productivity": "⚡",
    "audio": "🎵", "music": "🎵", "podcast": "🎙️", "data": "📈",
    "analytics": "📈", "design": "🖌️", "finance": "💰", "seo": "🔎",
    "social": "📱", "sales": "💼", "email": "📧", "health": "🏥",
    "education": "📚", "legal": "⚖️", "security": "🔒",
}
SOURCE_LABELS = {
    "producthunt":  ("PH",      "#da552f"),
    "taaft":        ("TAAFT",   "#6c5ce7"),
    "huggingface":  ("HF",      "#ff9500"),
    "futurepedia":  ("FP",      "#00b894"),
    "awesomelist":  ("AWESOME", "#0984e3"),
    "futuretools":  ("FT",      "#e17055"),
}
EXAMPLES = [
    "generate AI videos for YouTube",
    "write better marketing emails",
    "customer support chatbot",
    "transcribe meetings to text",
    "SEO keyword research",
    "generate images from text",
    "code review assistant",
    "create pitch deck for startup",
    "translate website content",
    "automate HR recruiting",
    "AI music generation",
    "summarize long documents",
]
PRICING_COLORS = {
    "free":     ("#d1fae5", "#065f46"),
    "freemium": ("#dbeafe", "#1e40af"),
    "paid":     ("#fef3c7", "#92400e"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cat_emoji(cat):
    c = (cat or "").lower()
    for k, e in CATEGORY_EMOJIS.items():
        if k in c:
            return e
    return "🤖"


def src_badge(source):
    s = (source or "").lower().split("+")[0].strip()
    label, color = SOURCE_LABELS.get(s, ("AI", "#636e72"))
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:0.67em;font-weight:700;'
            f'letter-spacing:0.03em">{label}</span>')


def pricing_badge(tier):
    t = (tier or "unknown").lower()
    for key in ("free", "freemium", "paid"):
        if key in t:
            bg, fg = PRICING_COLORS[key]
            label = key.capitalize()
            return (f'<span style="background:{bg};color:{fg};padding:2px 8px;'
                    f'border-radius:4px;font-size:0.67em;font-weight:600">{label}</span>')
    return ""


def match_pill(score):
    if score >= 0.50:   bg, fg = "#d1fae5", "#065f46"
    elif score >= 0.38: bg, fg = "#dbeafe", "#1e40af"
    elif score >= 0.28: bg, fg = "#fef3c7", "#92400e"
    else:               bg, fg = "#f1f5f9", "#64748b"
    return (f'<span style="background:{bg};color:{fg};padding:3px 10px;'
            f'border-radius:20px;font-size:0.73em;font-weight:600">'
            f'Score {score:.2f}</span>')


def score_bar(label, val, color):
    pct = min(int(val * 100), 100)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
        f'<span style="font-size:0.68em;color:#94a3b8;width:56px;text-align:right">{label}</span>'
        f'<div style="flex:1;height:4px;border-radius:2px;background:#e2e8f0">'
        f'<div style="width:{pct}%;height:4px;border-radius:2px;background:{color}"></div></div>'
        f'<span style="font-size:0.68em;color:#94a3b8;width:30px">{val:.2f}</span>'
        f'</div>'
    )


def cat_tags(cat):
    tags = [t.strip() for t in (cat or "").split(",") if t.strip()][:3]
    return "".join(
        f'<span style="background:#eff6ff;color:#3b82f6;padding:2px 8px;'
        f'border-radius:10px;font-size:0.7em;margin-right:4px;'
        f'border:1px solid #bfdbfe">{t}</span>'
        for t in tags
    )


def call_search(query, top_k=10):
    boosts = st.session_state.get("category_boost", {})
    for attempt in range(2):
        try:
            r = requests.post(
                f"{API_URL}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "category_boosts": boosts if boosts else None,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            st.error("⚠️ API not running — start it: `python run_phase09.py`")
            return None
        except requests.exceptions.ReadTimeout:
            if attempt == 0:
                st.toast("⏳ Model loading — retrying…")
                time.sleep(3)
                continue
            st.error("⚠️ Request timed out. Try again.")
            return None
        except Exception as e:
            st.error(f"API error: {e}")
            return None
    return None


def category_match_badge(match_type: str) -> str:
    if match_type == "exact":
        return ('<span style="background:#d1fae5;color:#065f46;padding:2px 7px;'
                'border-radius:8px;font-size:0.65em;font-weight:600">✓ domain</span>')
    if match_type == "adjacent":
        return ('<span style="background:#fef3c7;color:#92400e;padding:2px 7px;'
                'border-radius:8px;font-size:0.65em;font-weight:600">~ adjacent</span>')
    return ""


def render_card(rank, tool):
    url          = tool.get("url", "")
    s_sem        = tool.get("semantic_score", 0)
    s_bm25       = tool.get("bm25_score", 0)
    s_cat        = tool.get("category_score", 0)
    s_final      = tool.get("final_score", 0)
    desc         = (tool.get("description") or "")[:220]
    name         = tool.get("tool_name", "Unknown")
    pricing      = tool.get("pricing_tier", "")
    source       = tool.get("source", "")
    cat          = tool.get("category", "")
    cat_match    = tool.get("category_match", "none")

    # Rank accent colours: gold / silver / bronze / slate
    accent = {0: "#f59e0b", 1: "#94a3b8", 2: "#b45309"}.get(rank, "#e2e8f0")
    rank_bg = {0: "#f59e0b", 1: "#94a3b8", 2: "#b45309"}.get(rank, "#cbd5e1")

    link_html = (
        f'<a href="{url}" target="_blank" style="color:#1e293b;font-weight:700;'
        f'font-size:1em;text-decoration:none">'
        f'{cat_emoji(cat)}&nbsp;{name}&nbsp;↗</a>'
        if url else
        f'<span style="color:#1e293b;font-weight:700;font-size:1em">'
        f'{cat_emoji(cat)}&nbsp;{name}</span>'
    )

    badges = src_badge(source)
    p_badge = pricing_badge(pricing)
    if p_badge:
        badges += f'&nbsp;{p_badge}'
    cm_badge = category_match_badge(cat_match)
    if cm_badge:
        badges += f'&nbsp;{cm_badge}'

    feedback_given = st.session_state["feedback"].get(name, "")

    st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
     padding:18px 20px 14px 20px;border-left:4px solid {accent};
     box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:4px">

  <!-- Header row -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex:1;min-width:0">
      <span style="background:{rank_bg};color:#fff;border-radius:50%;
        width:22px;height:22px;display:inline-flex;align-items:center;
        justify-content:center;font-size:.68em;font-weight:800;
        flex-shrink:0">{'#' if rank > 2 else rank+1}</span>
      {link_html}
    </div>
    <div style="flex-shrink:0;margin-left:10px">{match_pill(s_final)}</div>
  </div>

  <!-- Badges row -->
  <div style="margin-bottom:8px;display:flex;gap:4px;flex-wrap:wrap">
    {badges}&nbsp;{cat_tags(cat)}
  </div>

  <!-- Description -->
  <p style="color:#4b5563;font-size:0.855em;line-height:1.65;margin:0 0 12px 0">
    {desc}{"…" if len(tool.get("description",""))>220 else ""}
  </p>

  <!-- Score bars -->
  <div style="border-top:1px solid #f1f5f9;padding-top:8px">
    {score_bar("semantic", s_sem, "#4361ee")}
    {score_bar("BM25",     s_bm25, "#7c3aed")}
    {score_bar("category", s_cat,  "#10b981")}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Feedback row (outside the card div to use Streamlit columns) ──────────
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
    up_style   = "🟢 👍" if feedback_given == "up"   else "👍"
    down_style = "🔴 👎" if feedback_given == "down" else "👎"
    with fb_col1:
        if st.button(up_style,   key=f"up_{name}_{rank}",   help="Relevant — teach ToolSense"):
            record_feedback(name, cat, "up")
            st.toast(f"✅ Noted! Boosting **{cat}** results for this session.", icon="✅")
            st.rerun()
    with fb_col2:
        if st.button(down_style, key=f"dn_{name}_{rank}",   help="Not relevant — teach ToolSense"):
            record_feedback(name, cat, "down")
            st.toast(f"👎 Got it. Reducing **{cat}** weight for this session.", icon="👎")
            st.rerun()
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToolSense",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* hide Streamlit chrome */
  #MainMenu, footer { visibility:hidden !important; }

  /* constrain content width without hiding sidebar */
  .block-container {
      padding-top: 1.5rem !important;
      padding-bottom: 2rem !important;
      max-width: 980px !important;
  }

  /* search input */
  .stTextInput > div > div > input {
      border-radius: 12px !important;
      padding: 13px 18px !important;
      font-size: 1.02em !important;
      border: 2px solid #e2e8f0 !important;
      box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
  }
  .stTextInput > div > div > input:focus {
      border-color: #4361ee !important;
      box-shadow: 0 0 0 3px rgba(67,97,238,.12) !important;
  }

  /* primary button */
  .stButton > button[kind="primary"] {
      border-radius: 12px !important;
      font-size: 0.97em !important;
      padding: 13px 20px !important;
      background: linear-gradient(135deg,#4361ee,#7c3aed) !important;
      border: none !important;
  }

  /* example chip buttons */
  div[data-testid="column"] .stButton > button {
      background: #f8fafc !important;
      color: #475569 !important;
      border: 1px solid #e2e8f0 !important;
      border-radius: 20px !important;
      font-size: 0.78em !important;
      padding: 5px 10px !important;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
  }
  div[data-testid="column"] .stButton > button:hover {
      background: #eff6ff !important;
      color: #3b82f6 !important;
      border-color: #bfdbfe !important;
  }

  /* divider */
  hr { border-color: #e2e8f0 !important; margin: 10px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "feedback" not in st.session_state:
    # {tool_name: "up" | "down"}
    st.session_state["feedback"] = {}
if "category_boost" not in st.session_state:
    # Session-scoped category adjustments from positive feedback
    # {category_keyword: boost_count}
    st.session_state["category_boost"] = {}


def record_feedback(tool_name: str, category: str, vote: str) -> None:
    """
    Store thumbs up/down for a tool in session state.
    Positive votes boost the tool's category for subsequent queries this session.
    Logs to W&B via the API if available (fire-and-forget, never blocks UI).
    """
    prev = st.session_state["feedback"].get(tool_name)
    if prev == vote:
        # Clicking same button again → undo
        del st.session_state["feedback"][tool_name]
        return

    st.session_state["feedback"][tool_name] = vote

    # Update session category boost
    for part in (category or "").split(","):
        for w in part.lower().strip().split():
            if w:
                delta = 1 if vote == "up" else -1
                st.session_state["category_boost"][w] = (
                    st.session_state["category_boost"].get(w, 0) + delta
                )

    # Log to W&B via API (best-effort)
    try:
        requests.post(
            f"{API_URL}/feedback",
            json={"tool_name": tool_name, "vote": vote, "category": category},
            timeout=2,
        )
    except Exception:
        pass  # W&B logging is optional — never break the UI


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 ToolSense")
    st.caption("Hybrid SBERT + BM25 recommender · 2,121 AI tools")
    st.divider()

    # Results slider + clear history
    top_k = st.slider("Results to show", min_value=3, max_value=20, value=8, step=1)
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    st.divider()

    # Example queries
    st.markdown("**💡 Example searches**")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["run_query"] = ex
            st.rerun()

    st.divider()

    # Query type legend
    st.markdown("**Dynamic weight mode**")
    for qt, (label, bg, fg, desc) in _QUERY_TYPE_META.items():
        st.markdown(
            f'<div style="background:{bg};color:{fg};padding:5px 10px;'
            f'border-radius:8px;font-size:0.78em;margin-bottom:4px">'
            f'<strong>{label}</strong><br><span style="opacity:0.8">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Session feedback summary
    fb = st.session_state.get("feedback", {})
    boosts = st.session_state.get("category_boost", {})
    if fb:
        st.markdown("**Session feedback**")
        ups   = [t for t, v in fb.items() if v == "up"]
        downs = [t for t, v in fb.items() if v == "down"]
        if ups:
            st.markdown(f"👍 **{len(ups)}** relevant: " + ", ".join(f"`{t}`" for t in ups[:3])
                        + ("…" if len(ups) > 3 else ""))
        if downs:
            st.markdown(f"👎 **{len(downs)}** not relevant: " + ", ".join(f"`{t}`" for t in downs[:3])
                        + ("…" if len(downs) > 3 else ""))
        if boosts:
            active = {k: v for k, v in boosts.items() if v != 0}
            if active:
                st.caption("Active category adjustments: " +
                           ", ".join(f"{k} ({'+' if v>0 else ''}{v*10:.0f}%)" for k, v in active.items()))
        if st.button("Reset feedback", use_container_width=True):
            st.session_state["feedback"] = {}
            st.session_state["category_boost"] = {}
            st.rerun()
    else:
        st.caption("No feedback yet — use 👍/👎 on results to adapt rankings.")

    st.divider()
    st.markdown("**Scoring weights** *(Optuna-tuned)*")
    st.markdown("""
<div style="font-size:0.8em;color:#475569;line-height:1.8">
  SBERT semantic &nbsp;&nbsp;<code>0.41</code><br>
  BM25 keyword &nbsp;&nbsp;&nbsp;&nbsp;<code>0.45</code><br>
  Category boost &nbsp;&nbsp;<code>0.14</code>
</div>""", unsafe_allow_html=True)


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 60%,#0f172a 100%);
     padding:32px 36px 28px;border-radius:16px;margin-bottom:24px;
     box-shadow:0 4px 24px rgba(67,97,238,.18)">
  <h1 style="color:#fff;font-size:2em;font-weight:800;margin:0 0 6px 0;
             letter-spacing:-0.02em">🔍 ToolSense</h1>
  <p style="color:#94a3b8;margin:0;font-size:0.92em;line-height:1.5">
    Find the right AI tool — <strong style="color:#818cf8">2,121 tools</strong>
    indexed across ProductHunt, HuggingFace, AwesomeList &amp; more.<br>
    <span style="color:#64748b;font-size:0.88em">
      Powered by SBERT semantic search + BM25 keyword matching.
    </span>
  </p>
</div>
""", unsafe_allow_html=True)


# ── Search bar ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("run_query", "")
col_in, col_btn = st.columns([5, 1])
with col_in:
    query = st.text_input(
        "Search",
        value=prefill,
        placeholder="e.g. 'turn my podcast into short video clips'",
        label_visibility="collapsed",
        key="q_input",
    )
with col_btn:
    clicked = st.button("Search 🔍", use_container_width=True, type="primary")

st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

# ── Trigger search ────────────────────────────────────────────────────────────
trigger = (clicked or query.strip() != st.session_state["last_query"]) and query.strip()
if trigger:
    st.session_state["last_query"] = query.strip()
    with st.spinner("Finding best tools…"):
        t0   = time.perf_counter()
        data = call_search(query.strip(), top_k=20)   # always fetch 20, filter client-side
        ms   = (time.perf_counter() - t0) * 1000
    if data:
        st.session_state["history"].insert(0, {
            "query": query.strip(),
            "data":  data,
            "ms":    ms,
        })


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state["history"]:
    for h_idx, entry in enumerate(st.session_state["history"]):
        all_results = entry["data"].get("results", [])
        corrected   = entry["data"].get("corrected_query", "")

        results = all_results[:top_k]

        # Query header with query-type badge
        corr_html = ""
        if corrected and corrected.lower() != entry["query"].lower():
            corr_html = (
                f' <span style="color:#f59e0b;font-size:0.8em">'
                f'→ corrected to <em>"{corrected}"</em></span>'
            )

        qtype = _detect_query_type(entry["query"])
        qt_label, qt_bg, qt_fg, qt_desc = _QUERY_TYPE_META[qtype]
        qt_badge = (
            f'<span style="background:{qt_bg};color:{qt_fg};padding:2px 8px;'
            f'border-radius:6px;font-size:0.72em;font-weight:600;'
            f'margin-left:8px" title="{qt_desc}">{qt_label}</span>'
        )

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:14px">'
            f'<p style="font-size:1.05em;font-weight:700;color:#1e293b;margin:0">'
            f'🔎 &ldquo;{entry["query"]}&rdquo;{corr_html}{qt_badge}</p>'
            f'<p style="color:#94a3b8;font-size:0.8em;margin:0;white-space:nowrap">'
            f'{len(results)} tools&nbsp;·&nbsp;{entry["ms"]:.0f}&nbsp;ms</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not results:
            st.info("No tools found. Try a different search or fewer results.")
        else:
            for row in range(0, len(results), 2):
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    render_card(row, results[row])
                with c2:
                    if row + 1 < len(results):
                        render_card(row + 1, results[row + 1])

        if h_idx < len(st.session_state["history"]) - 1:
            st.divider()

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center;padding:60px 20px 40px">
  <div style="font-size:3.8em;margin-bottom:14px">🤖</div>
  <h3 style="color:#374151;font-weight:700;margin:0 0 8px 0;font-size:1.3em">
    What AI tool are you looking for?
  </h3>
  <p style="color:#94a3b8;font-size:0.92em;margin:0 0 28px 0;max-width:480px;
            margin-left:auto;margin-right:auto">
    Type a description in plain English. ToolSense uses semantic search + keyword
    matching to find the best match across 2,121 indexed tools.
  </p>
  <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;
              max-width:580px;margin:0 auto">
    <span style="background:#eff6ff;color:#3b82f6;padding:8px 16px;
      border-radius:20px;font-size:0.85em;border:1px solid #bfdbfe">🎬 Video generation</span>
    <span style="background:#f0fdf4;color:#16a34a;padding:8px 16px;
      border-radius:20px;font-size:0.85em;border:1px solid #bbf7d0">&#9997;&#65039; AI writing</span>
    <span style="background:#fdf4ff;color:#9333ea;padding:8px 16px;
      border-radius:20px;font-size:0.85em;border:1px solid #e9d5ff">🎨 Image tools</span>
    <span style="background:#fff7ed;color:#ea580c;padding:8px 16px;
      border-radius:20px;font-size:0.85em;border:1px solid #fed7aa">💻 Code assistants</span>
    <span style="background:#fefce8;color:#ca8a04;padding:8px 16px;
      border-radius:20px;font-size:0.85em;border:1px solid #fde68a">📢 Marketing</span>
    <span style="background:#f5f3ff;color:#7c3aed;padding:8px 16px;
      border-radius:20px;font-size:0.85em;border:1px solid #ddd6fe">🎵 Audio &amp; Music</span>
  </div>
</div>
""", unsafe_allow_html=True)
