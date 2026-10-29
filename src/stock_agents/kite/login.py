"""Run once each trading day to refresh KITE_ACCESS_TOKEN.

Kite's access token expires daily and requires a real browser login - this is
not automated here on purpose. Paste the resulting access_token into .env.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from kiteconnect import KiteConnect


def main() -> None:
    load_dotenv()
    api_key = os.environ["KITE_API_KEY"]
    api_secret = os.environ["KITE_API_SECRET"]

    kite = KiteConnect(api_key=api_key)
    print("Open this URL, log in, and paste the `request_token` from the redirect URL:\n")
    print(kite.login_url())

    request_token = input("\nrequest_token: ").strip()
    session = kite.generate_session(request_token, api_secret=api_secret)

    print("\nAccess token (valid until the next trading day):")
    print(session["access_token"])
    print("\nPut this in .env as KITE_ACCESS_TOKEN before running the runner.")


if __name__ == "__main__":
    main()
