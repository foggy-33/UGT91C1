"""简单口令门：给「实验室局域网共享部署」用的一道组内共享口令。

口令写在 ``.streamlit/secrets.toml`` 的 ``app_password`` 字段里（该文件已被
``.gitignore`` 忽略，不会进仓库）。未配置口令时直接放行，方便本地开发与测试。

定位说明：这是「防止组外人员误入」的轻量门禁，不是强安全边界。真正的安全
边界建议交给反向代理（basic auth）或内网访问控制。

用法：在 ``streamlit_app.py`` 中调用 ``require_password()``，未通过校验时会
渲染登录页并中断本次运行。
"""

from __future__ import annotations

import hmac

import streamlit as st

AUTH_KEY = "_ugt_authed"
INPUT_KEY = "_ugt_password_input"


def configured_password() -> str:
    """读取组内共享口令；未配置（或没有 secrets.toml）时返回空串。"""
    try:
        value = st.secrets.get("app_password", "")
    except Exception:  # 未创建 secrets.toml 时 st.secrets 会抛异常
        return ""
    return str(value).strip()


def require_password() -> None:
    """校验口令；未通过则渲染登录页并 ``st.stop()``。"""
    password = configured_password()
    if not password or st.session_state.get(AUTH_KEY):
        return

    st.title("UGT91C1 突变体活性预测平台")
    st.caption("本平台运行在实验室内部网络，请输入组内共享口令后进入。")

    with st.form("ugt_login"):
        entered = st.text_input("访问口令", type="password", key=INPUT_KEY)
        submitted = st.form_submit_button("进入平台", type="primary")

    if submitted:
        if hmac.compare_digest(entered, password):
            st.session_state[AUTH_KEY] = True
            st.rerun()
        else:
            st.error("口令不正确，请重新输入。")

    st.stop()
