"""Renew the Dhan token and store the new one as the vault's secret.  python -m vaaydo.dhan.renew

Dhan's RenewToken expires the current token and returns a new one, so the new token is written back
(GitHub Actions secret DHAN_ACCESS_TOKEN of the vault, via `gh` with VAULT_SECRETS_PAT) before anything
else, masked in the log, and handed to the rest of the job through $GITHUB_ENV. The PIN and TOTP secret
are never needed or stored. If this ever fails after the renewal, generate one token on Dhan Web and
save it as DHAN_ACCESS_TOKEN; the next run carries on from it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from .client import Dhan


def store(token: str, repo: str) -> None:
    err = ""
    for k in range(5):
        r = subprocess.run(["gh", "secret", "set", "DHAN_ACCESS_TOKEN", "--repo", repo], input=token, text=True,
                           capture_output=True, env={**os.environ, "GH_TOKEN": os.environ["VAULT_SECRETS_PAT"]})
        if r.returncode == 0:
            return
        err = (r.stderr or r.stdout).strip()[:200]
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"could not store the renewed token ({err}). The old token is dead: "
                       "generate one on Dhan Web and save it as DHAN_ACCESS_TOKEN.")


def main() -> int:
    repo = os.environ["VAULT_REPO"]
    dh = Dhan()
    v = dh.profile().get("tokenValidity")
    new = dh.renew()
    print(f"::add-mask::{new}", flush=True)                  # never shown in the log from here on
    store(new, repo)
    env = os.environ.get("GITHUB_ENV")
    if env:
        with open(env, "a") as f:
            f.write(f"DHAN_ACCESS_TOKEN={new}\n")
    after = Dhan(token=new, client_id=dh.client_id).profile()
    print(f"token renewed: was valid to {v}, now valid to {after.get('tokenValidity')}; "
          f"data plan {after.get('dataPlan')} to {after.get('dataValidity')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
