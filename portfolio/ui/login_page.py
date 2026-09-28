"""
Login/signup gate shown before any portfolio data is accessible.
"""

import streamlit as st

from portfolio.storage import auth, db


def render_signup_form():
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Username")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")

    if not submitted:
        return
    if password != password_confirm:
        st.error("Passwords don't match.")
        return
    try:
        auth.sign_up(username, name, email, password)
    except ValueError as e:
        st.error(str(e))
        return
    st.success("Account created! Switch to the Log in tab to sign in.")


def require_login():
    """Show the login/signup page and halt the script until the user is authenticated."""
    try:
        db.ensure_schema()
    except Exception as e:
        st.error(f"Could not connect to the database: {e}")
        st.stop()

    authenticator = auth.get_authenticator()

    if st.session_state.get("authentication_status") is not True:
        st.title("📊 Portfolio Backtester")
        st.markdown("Build your pie and see how it would have performed!")

        login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
        with login_tab:
            authenticator.login(location="main")
            if st.session_state.get("authentication_status") is False:
                st.error("Username or password is incorrect")
        with signup_tab:
            render_signup_form()
        st.stop()

    if "user_id" not in st.session_state:
        st.session_state.user_id = auth.get_user_id(st.session_state["username"])

    return authenticator
