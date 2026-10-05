"""Alert delivery: Prometheus -> Alertmanager -> local webhook receiver.

The receiver is tested directly (formatting and a real HTTP POST); the YAML
is checked for the wiring that would otherwise only fail inside Docker."""

import json
import sys
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from test_api import FakeEngine

from egyptian_civil_code_rag.api import create_app

REPO = Path(__file__).parent.parent
MON = REPO / "deploy" / "monitoring"
sys.path.insert(0, str(MON / "alert-receiver"))

import receiver  # noqa: E402

PAYLOAD = {
    "status": "firing",
    "alerts": [
        {
            "status": "firing",
            "labels": {"alertname": "RagFaithfulnessLow", "severity": "critical"},
            "annotations": {"summary": "RAGAS faithfulness 0.62 is below 0.80"},
        },
        {
            "status": "resolved",
            "labels": {"alertname": "RagQueryDrift", "severity": "warning", "window": "off_topic"},
            "annotations": {"summary": "off-corpus share high"},
        },
    ],
}


def test_one_line_per_alert_with_status_severity_and_summary():
    lines = receiver.format_notification(PAYLOAD)
    assert len(lines) == 2
    assert "FIRING critical RagFaithfulnessLow: RAGAS faithfulness 0.62" in lines[0]
    assert "RESOLVED warning RagQueryDrift window=off_topic: off-corpus share high" in lines[1]


def test_empty_payload_gives_no_lines():
    assert receiver.format_notification({}) == []


@pytest.fixture
def server(tmp_path, monkeypatch):
    log = tmp_path / "logs" / "alerts.log"
    monkeypatch.setattr(receiver, "LOG_PATH", str(log))
    srv = HTTPServer(("127.0.0.1", 0), receiver.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/alerts", log
    srv.shutdown()


def post(url, body: bytes):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def test_webhook_post_is_written_to_the_log(server):
    url, log = server
    assert post(url, json.dumps(PAYLOAD).encode()) == 200
    text = log.read_text(encoding="utf-8")
    assert "FIRING critical RagFaithfulnessLow" in text and "RESOLVED" in text


def test_bad_json_is_rejected_and_not_logged(server):
    url, log = server
    assert post(url, b"not json") == 400
    assert not log.exists()


def load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_wiring_matches_compose_service_names():
    compose = load(MON / "docker-compose.monitoring.yml")["services"]
    am = load(MON / "alertmanager" / "alertmanager.yml")
    prom = load(MON / "prometheus" / "prometheus.yml")

    # Prometheus -> Alertmanager
    targets = prom["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"]
    assert targets == ["alertmanager:9093"] and "alertmanager" in compose

    # Alertmanager -> receiver, for every receiver the routes use
    hooks = {r["name"]: r["webhook_configs"][0]["url"] for r in am["receivers"]}
    used = {am["route"]["receiver"]} | {r["receiver"] for r in am["route"].get("routes", [])}
    assert used <= set(hooks)
    for url in hooks.values():
        assert url == "http://alert-receiver:8080/alerts"
    assert compose["alert-receiver"]["command"][-1] == "8080"


def test_demo_report_drives_the_faithfulness_alert(monkeypatch):
    # RAGAS_REPORT swaps the report the API exports; the demo file is below
    # the 0.80 alert threshold, so RagFaithfulnessLow fires on first evaluation.
    demo = MON / "demo" / "ragas_low_faithfulness.json"
    monkeypatch.setenv("RAGAS_REPORT", str(demo))
    text = TestClient(create_app(engine=FakeEngine())).get("/metrics").text
    line = next(
        ln
        for ln in text.splitlines()
        if ln.startswith('rag_ragas_faithfulness{source="ragas_low_faithfulness.json"}')
    )
    value = float(line.split()[-1])
    assert value == pytest.approx((0.55 + 0.60 + 0.71) / 3)

    rules = load(MON / "prometheus" / "alerts.yml")
    expr = next(
        r["expr"]
        for g in rules["groups"]
        for r in g["rules"]
        if r.get("alert") == "RagFaithfulnessLow"
    )
    assert value < float(expr.split("<")[-1])
