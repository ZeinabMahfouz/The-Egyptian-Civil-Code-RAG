"""Minimal Alertmanager webhook receiver: one log line per notification.

Stands in for email/Slack so alert delivery can be demonstrated without any
account. Lines go to stdout (docker compose logs alert-receiver) and to
/logs/alerts.log (deploy/monitoring/alert-receiver/logs/ on the host).
Standard library only.
"""

import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_PATH = os.environ.get("ALERT_LOG", "/logs/alerts.log")


def format_notification(payload: dict) -> list[str]:
    """Alertmanager webhook JSON -> one human-readable line per alert."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    for alert in payload.get("alerts", []):
        labels = alert.get("labels", {})
        ann = alert.get("annotations", {})
        status = alert.get("status", payload.get("status", "?")).upper()
        extra = f" window={labels['window']}" if "window" in labels else ""
        lines.append(
            f"[{now}] {status} {labels.get('severity', '-')} "
            f"{labels.get('alertname', '?')}{extra}: {ann.get('summary', '')}"
        )
    return lines


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server naming)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            lines = format_notification(json.loads(body))
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        for line in lines:
            print(line, flush=True)
        if lines and LOG_PATH:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # keep stdout to the alert lines only
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"alert receiver listening on :{port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
