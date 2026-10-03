"""Load test for the /ask API: 50 concurrent users asking real questions.

Run (see notebooks/kaggle_load_test.ipynb for the full setup):

    locust -f load_test/locustfile.py --host http://localhost:8001 \\
        --headless -u 50 -r 5 -t 5m --csv reports/locust_u50 --html reports/locust_u50.html

Each simulated user picks a question from the 54-question evaluation set
(Arabic and English, all four categories, including out-of-corpus ones) and
waits 0.5-2 s between requests, roughly someone reading an answer before
asking the next question. A request counts as failed if it isn't HTTP 200 or
the body isn't a usable answer (an empty answer is a failure even with 200).
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    q["question"]
    for q in json.loads(
        (
            Path(__file__).resolve().parent.parent / "tests" / "eval" / "eval_questions.json"
        ).read_text(encoding="utf-8")
    )["questions"]
]


class LegalQuestionUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(20)
    def ask(self):
        question = random.choice(QUESTIONS)
        with self.client.post(
            "/ask", json={"question": question}, name="/ask", catch_response=True, timeout=180
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("response is not JSON")
                return
            if not body.get("answer", "").strip():
                resp.failure("empty answer")

    @task(1)
    def health(self):
        # what a load balancer would poll; should stay fast under load
        self.client.get("/health", name="/health")
