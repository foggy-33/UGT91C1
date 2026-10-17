"""平台主题：蓝黑金「白天 / 夜间」两套配色、原生主题切换与美化 CSS。

两层机制：
  1. ``apply_native_theme()`` —— 把当前主题写进 Streamlit 的运行时主题配置，
     让内置组件（表格、上传框、下拉、提示条等）自己换成对应的底色与文字色。
     这是夜间模式可读性的关键：只注入 CSS 无法改变内置组件的文字色。
  2. ``inject_css()`` —— 注入品牌配色变量与美化样式（指标卡阴影、小标题竖线、
     圆角表格、段落行距等），并对文字色兜底，即使原生主题不可用也保证可读。

颜色只走蓝 / 黑 / 金，不使用红色系。
"""

from __future__ import annotations

import altair as alt
import streamlit as st

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "base": "light",
        "bg": "#f5f8fc",
        "panel": "#ffffff",
        "panel2": "#eaf0f8",
        "text": "#0f1720",
        "muted": "#56657e",
        "border": "#d9e3ef",
        "blue": "#1f4e9c",
        "blue_light": "#6b8cbf",
        "gold": "#a9822f",
        "grid": "#e4eaf3",
        "code_bg": "#f1f5fa",
        "shadow": "0 1px 2px rgba(15, 23, 32, 0.05)",
    },
    "dark": {
        "base": "dark",
        "bg": "#0b1220",
        "panel": "#16233a",
        "panel2": "#1b2b45",
        "text": "#e9eefb",
        "muted": "#a3b2cb",
        "border": "#2b3d5c",
        "blue": "#5b9bff",
        "blue_light": "#8fb4f2",
        "gold": "#dcbb72",
        "grid": "#243454",
        "code_bg": "#101a2c",
        "shadow": "0 1px 2px rgba(0, 0, 0, 0.35)",
    },
}


def theme_name() -> str:
    """当前主题名（``light`` / ``dark``）。"""
    return st.session_state.get("theme", "light")


def palette() -> dict[str, str]:
    """当前主题调色板。"""
    return THEMES[theme_name()]


def apply_native_theme() -> None:
    """把当前主题同步到 Streamlit 原生主题配置；切换时自动触发一次重跑。

    主题配置是进程级的，本平台为单用户本地工具，不存在多用户争用问题。
    """
    name = theme_name()
    if st.session_state.get("_applied_theme") == name:
        return

    p = THEMES[name]
    options = {
        "theme.base": p["base"],
        "theme.primaryColor": p["blue"],
        "theme.backgroundColor": p["bg"],
        "theme.secondaryBackgroundColor": p["panel"],
        "theme.textColor": p["text"],
        "theme.linkColor": p["blue"],
        "theme.borderColor": p["border"],
    }
    try:
        for key, value in options.items():
            st._config.set_option(key, value)
    except Exception:  # 私有接口不可用时退化为纯 CSS 方案
        return

    st.session_state["_applied_theme"] = name
    st.rerun()


def inject_css() -> None:
    """注入品牌配色变量与美化样式（色值走 CSS 变量，两套主题共用一套规则）。"""
    p = palette()
    is_dark = theme_name() == "dark"

    # 夜间模式额外做一层「文字强制提亮」兜底：即使原生主题没能覆盖某个内置组件，
    # 也不会出现深底深字的不可读情况。浅色模式不动这些，保留组件本来的层次。
    dark_text_fix = (
        """
.stApp p, .stApp span, .stApp li, .stApp strong, .stApp em, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp td, .stApp th, .stApp caption { color: var(--ugt-text) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--ugt-muted) !important; }
[data-testid="stMetricValue"] { color: var(--ugt-blue) !important; }
[data-testid="stMetricLabel"] { color: var(--ugt-muted) !important; }
[data-testid="stSidebarNav"] span { color: var(--ugt-text) !important; }
[data-testid="stMarkdownContainer"] a, [data-testid="stMarkdownContainer"] a span { color: var(--ugt-blue) !important; }
code, pre { color: var(--ugt-gold) !important; }
"""
        if is_dark
        else """
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--ugt-muted) !important; }
[data-testid="stMetricValue"] { color: var(--ugt-blue) !important; }
[data-testid="stMetricLabel"] { color: var(--ugt-muted) !important; }
code, pre { color: var(--ugt-gold) !important; background: var(--ugt-code-bg) !important; }
"""
    )

    st.markdown(
        f"""
<style>
:root {{
    --ugt-bg: {p['bg']};
    --ugt-panel: {p['panel']};
    --ugt-panel2: {p['panel2']};
    --ugt-text: {p['text']};
    --ugt-muted: {p['muted']};
    --ugt-border: {p['border']};
    --ugt-blue: {p['blue']};
    --ugt-blue-light: {p['blue_light']};
    --ugt-gold: {p['gold']};
    --ugt-grid: {p['grid']};
    --ugt-code-bg: {p['code_bg']};
    --ugt-shadow: {p['shadow']};
}}

/* ---------- 基础版式 ---------- */
.stApp {{ background: var(--ugt-bg); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 3.2rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ line-height: 1.8; }}

{dark_text_fix}

/* ---------- 说明提示块（替代内联样式，保证两种主题下都随变量变色） ---------- */
.ugt-callout {{
    border-left: 4px solid var(--ugt-blue);
    background: var(--ugt-panel2);
    border-radius: 0 10px 10px 0;
    margin: 8px 0 6px;
    padding: 14px 18px;
    line-height: 1.75;
}}
.ugt-callout strong {{ color: var(--ugt-blue) !important; }}

/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"] {{
    background: var(--ugt-panel);
    border-right: 1px solid var(--ugt-border);
}}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{ border-radius: 8px; }}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{ background: var(--ugt-panel2); }}

/* ---------- 指标卡 ---------- */
[data-testid="stMetric"] {{
    background: var(--ugt-panel);
    border: 1px solid var(--ugt-border);
    border-radius: 12px;
    box-shadow: var(--ugt-shadow);
    padding: 12px 16px;
}}

/* ---------- 小标题：左侧蓝色竖线 ---------- */
.stApp h3 {{
    border-left: 3px solid var(--ugt-blue);
    padding-left: 10px;
    letter-spacing: 0.2px;
}}

/* ---------- 表格 / 展开器 / 提示条 ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--ugt-border);
    border-radius: 10px;
    overflow: hidden;
}}
[data-testid="stExpander"] details {{
    background: var(--ugt-panel);
    border: 1px solid var(--ugt-border);
    border-radius: 10px;
}}
[data-testid="stExpander"] summary:hover {{ background: var(--ugt-panel2); }}
[data-testid="stAlert"] {{ border-radius: 10px; }}

/* ---------- 控件 ---------- */
.stApp button {{ border-radius: 8px; }}
[data-testid="stFileUploaderDropzone"] {{
    background: var(--ugt-panel);
    border: 1px dashed var(--ugt-border);
    border-radius: 10px;
}}

hr {{ border-color: var(--ugt-border); }}
</style>
""",
        unsafe_allow_html=True,
    )


def style_chart(chart):
    """给 Altair 图统一套上当前主题的背景 / 轴 / 图例配色。"""
    if chart is None:
        return None
    p = palette()
    return chart.configure(
        background=p["panel"],
        view=alt.ViewConfig(stroke=None),
        axis=alt.AxisConfig(
            labelColor=p["muted"],
            titleColor=p["text"],
            gridColor=p["grid"],
            domainColor=p["border"],
            tickColor=p["border"],
            labelFontSize=12,
            titleFontSize=12,
        ),
        legend=alt.LegendConfig(labelColor=p["muted"], titleColor=p["text"]),
        title=alt.TitleConfig(color=p["text"]),
    )
