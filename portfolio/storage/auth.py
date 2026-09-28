"""
Thin wrapper around streamlit-authenticator, isolating its API from app.py.

We only use the library for the login form and cookie/session handling
(Authenticate.login/.logout), which has been the stable part of its API.
Signup is a small custom form here instead of the library's register_user()
widget, since that widget's pre-authorized-email gating and parameter names
have shifted across versions — plain bcrypt hashing avoids that entirely and
is still exactly what Authenticate.login() verifies against.

Pinned to streamlit-authenticator==0.4.2 (requirements.txt) — this library's
API has changed materially across 0.2.x/0.3.x/0.4.x, so don't upgrade
casually.
"""

import logging

import bcrypt
import streamlit as st
import streamlit_authenticator as stauth

from . import db

logger = logging.getLogger(__name__)


def _fetch_credentials_dict() -> dict:
    rows = db.execute(
        "SELECT username, name, email, password_hash FROM users",
        fetch=True,
    )
    return {
        "usernames": {
            row["username"]: {
                "name": row["name"],
                "email": row["email"],
                "password": row["password_hash"],
            }
            for row in rows
        }
    }


fetch_credentials_dict = st.cache_resource(show_spinner=False)(_fetch_credentials_dict)


def get_authenticator() -> stauth.Authenticate:
    """Build an Authenticate object from the current set of users in the DB."""
    auth_cfg = st.secrets["auth"]
    credentials = fetch_credentials_dict()
    return stauth.Authenticate(
        credentials,
        auth_cfg["cookie_name"],
        auth_cfg["cookie_key"],
        int(auth_cfg.get("cookie_expiry_days", 30)),
    )


def username_exists(username: str) -> bool:
    rows = db.execute(
        "SELECT 1 FROM users WHERE username = %s",
        (username,),
        fetch=True,
    )
    return bool(rows)


def email_exists(email: str) -> bool:
    rows = db.execute(
        "SELECT 1 FROM users WHERE email = %s",
        (email,),
        fetch=True,
    )
    return bool(rows)


def sign_up(username: str, name: str, email: str, password: str) -> None:
    """
    Create a new user with a bcrypt-hashed password, then invalidate the
    credentials cache so the next authenticator rebuild (including this
    rerun) sees the new user.

    Raises ValueError on invalid or duplicate input.
    """
    username = username.strip().lower()
    name = name.strip()
    email = email.strip().lower()

    if not username or " " in username:
        raise ValueError("Username is required and can't contain spaces.")
    if not name:
        raise ValueError("Name is required.")
    if "@" not in email:
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if username_exists(username):
        raise ValueError(f"Username '{username}' is already taken.")
    if email_exists(email):
        raise ValueError(f"Email '{email}' is already registered.")

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    db.execute(
        """
        INSERT INTO users (username, name, email, password_hash)
        VALUES (%s, %s, %s, %s)
        """,
        (username, name, email, password_hash),
    )
    fetch_credentials_dict.clear()
    logger.info(f"Registered new user '{username}'")


def get_user_id(username: str) -> int:
    """Resolve a username to its DB user_id."""
    rows = db.execute(
        "SELECT id FROM users WHERE username = %s",
        (username,),
        fetch=True,
    )
    if not rows:
        raise ValueError(f"No user found with username '{username}'")
    return rows[0]["id"]
