"""Read the private vault (thebullishvalue/vaaydo-vault) from the app, with the read-only token in Streamlit's secrets.

    [vault]                      [auth]
    repo = "owner/name"          commodities_password = "..."
    branch = "main"
    token = "github_pat_..."     (fine-grained: Contents read-only on the vault only)

Dhan's data never enters the public repository: the app reads it here at run time, behind a login, and never
writes back. Nothing in this module calls Dhan.
"""
from __future__ import annotations

import hmac
import io

import pandas as pd
import requests


class VaultUnavailable(RuntimeError):
    pass


def _cfg(secrets) -> dict:
    try:
        v = secrets["vault"]
        return {"repo": v["repo"], "branch": v.get("branch", "main"), "token": v["token"]}
    except Exception:
        raise VaultUnavailable("The vault is not configured (Streamlit secrets [vault] repo / token).")


def fetch(path: str, secrets, timeout: int = 60) -> bytes:
    """One file's bytes from the vault (GitHub contents API, raw media type: files up to 100 MB)."""
    c = _cfg(secrets)
    r = requests.get(f"https://api.github.com/repos/{c['repo']}/contents/{path}", params={"ref": c["branch"]},
                     headers={"Authorization": f"Bearer {c['token']}", "Accept": "application/vnd.github.raw",
                              "X-GitHub-Api-Version": "2022-11-28"}, timeout=timeout)
    if r.status_code == 404:
        raise VaultUnavailable(f"{path} is not in the vault yet.")
    if r.status_code in (401, 403):
        raise VaultUnavailable("The vault token was refused (expired, or without Contents: read on the vault).")
    r.raise_for_status()
    return r.content


def parquet(path: str, secrets) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(fetch(path, secrets)))


def password_ok(entered: str, secrets) -> bool:
    try:
        want = str(secrets["auth"]["commodities_password"])
    except Exception:
        return False
    return bool(entered) and hmac.compare_digest(entered.encode(), want.encode())
