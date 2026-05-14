"""Gmail MCP server — exposes a send_email tool over stdio.

One-time setup: run `python gmail_mcp_server.py auth` to complete the
OAuth flow. The browser opens, you grant Gmail send scope, and the
refresh token is cached in token.json next to this file. After that,
the chatbot launches this script as a stdio subprocess.
"""

from __future__ import annotations

import base64
import sys
from email.message import EmailMessage
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mcp.server.fastmcp import FastMCP

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
HERE = Path(__file__).parent
CREDENTIALS_FILE = HERE / "credentials.json"
TOKEN_FILE = HERE / "token.json"


def get_gmail_service():
    creds: Credentials | None = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


mcp = FastMCP("gmail")


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email from the user's Gmail account. Returns the sent message id."""
    if not TOKEN_FILE.exists():
        return (
            "Error: Gmail not authorized yet. "
            "Run `python gmail_mcp_server.py auth` once, then retry."
        )
    service = get_gmail_service()
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"sent: {sent.get('id', '')}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        get_gmail_service()
        print(f"Auth complete. Token saved to {TOKEN_FILE}")
    else:
        mcp.run()
