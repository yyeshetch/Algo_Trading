from kiteconnect import KiteConnect
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import webbrowser


def load_credentials(creds_path: Path) -> dict:
    with creds_path.open() as f:
        return json.load(f)


def save_credentials(creds_path: Path, creds: dict) -> None:
    with creds_path.open("w") as f:
        json.dump(creds, f, indent=4)


def extract_request_token_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url.strip())
        qs = parse_qs(parsed.query or "")
        token_list = qs.get("request_token") or []
        if token_list:
            return token_list[0]
    except Exception:
        return None
    return None


def main():
    """
    Interactive helper to refresh Kite access_token.

    Steps:
      1. Reads api_key and api_secret from kite_trade/credentials.json
      2. Opens the Zerodha login URL:
           https://kite.zerodha.com/connect/login?v=3&api_key=<api_key>
      3. You log in and are redirected to a URL that contains ?request_token=...
      4. Paste that final redirected URL (or just the request_token) back here
      5. Script exchanges request_token -> access_token and updates credentials.json
    """
    base_dir = Path(__file__).resolve().parent
    creds_path = base_dir / "credentials.json"

    creds = load_credentials(creds_path)
    api_key = creds.get("api_key")
    api_secret = creds.get("api_secret")

    if not api_key or not api_secret:
        print("Error: 'api_key' and 'api_secret' must be present in credentials.json")
        return

    login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    print("Opening Zerodha login URL in your browser:")
    print(login_url)
    try:
        webbrowser.open(login_url)
    except Exception:
        # If browser can't be opened, user can still copy-paste the URL
        pass

    print()
    print("After logging in, you will be redirected to a URL like:")
    print("  https://your-redirect-url.com/?request_token=XXXX&action=login&status=success")
    print("Copy that entire URL and paste it below, or paste just the request_token value.")
    user_input = input("Paste redirected URL or request_token: ").strip()

    request_token = None
    # Try to parse as full URL first
    request_token = extract_request_token_from_url(user_input)
    if not request_token:
        # Assume user pasted the raw token
        request_token = user_input

    if not request_token:
        print("Error: Could not extract request_token.")
        return

    kite = KiteConnect(api_key=api_key)

    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as e:
        print(f"Error generating session: {e}")
        return

    access_token = data.get("access_token")
    if not access_token:
        print("Error: No access_token returned from generate_session")
        return

    creds["access_token"] = access_token
    save_credentials(creds_path, creds)

    print("New access_token saved to credentials.json")


if __name__ == "__main__":
    main()

