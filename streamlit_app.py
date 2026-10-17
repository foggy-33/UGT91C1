"""UGT91C1 突变体活性预测平台（Streamlit 入口）。

启动：

    streamlit run streamlit_app.py

页面逻辑放在 app_pages/，共享业务逻辑放在 prediction_core.py，
中英文文案放在 i18n.py，主题配色放在 theme.py，共享口令门放在 auth.py。
"""

import streamlit as st

from auth import require_password
from i18n import LANG_OPTIONS, lang, t
from theme import apply_native_theme, inject_css, theme_name

st.set_page_config(
    page_title="UGT91C1 突变体活性预测平台",
    page_icon=":material/biotech:",
    layout="wide",
)

# 全局状态：预测结果跨页面保留，语言与主题默认中文 + 白天。
st.session_state.setdefault("predict_result", None)
st.session_state.setdefault("lang", "zh")
st.session_state.setdefault("theme", "light")

# 先把主题同步给 Streamlit 原生配置（切换时会自动重跑一次），再渲染页面。
apply_native_theme()

inject_css()

# 组内共享口令门：放在主题之后、导航之前，未通过校验时只渲染登录页。
require_password()

# --------------------------------------------------------------------------- #
# 侧边导航栏顶部：中英文切换 + 白天 / 夜间切换
# 在 st.navigation 之前写入 sidebar，控件会排在导航菜单上方。
# --------------------------------------------------------------------------- #
st.session_state.setdefault("lang_ui", "中文" if lang() == "zh" else "English")
st.session_state.setdefault("theme_ui", theme_name() == "dark")

with st.sidebar:
    lang_choice = st.segmented_control(t("语言"), list(LANG_OPTIONS), key="lang_ui")
    is_dark = st.toggle(t("夜间模式"), key="theme_ui")

new_lang = LANG_OPTIONS.get(lang_choice or "中文", "zh")
new_theme = "dark" if is_dark else "light"
if new_lang != lang() or new_theme != theme_name():
    st.session_state["lang"] = new_lang
    st.session_state["theme"] = new_theme
    st.rerun()

# 导航顺序：「候选推荐」为默认首页，「平台说明」放最后（标题随语言动态生成）。
NAV = [
    ("app_pages/recommend.py", "候选推荐", ":material/rocket_launch:", True),
    ("app_pages/predict.py", "候选预测", ":material/science:", False),
    ("app_pages/ingest.py", "数据入库", ":material/database_upload:", False),
    ("app_pages/compare.py", "对比分析", ":material/compare_arrows:", False),
    ("app_pages/results.py", "结果浏览", ":material/table_chart:", False),
    ("app_pages/about.py", "平台说明", ":material/menu_book:", False),
]
pages = st.navigation(
    [st.Page(path, title=t(label), icon=icon, default=default) for path, label, icon, default in NAV],
    position="sidebar",
)

st.title(pages.title)
pages.run()
