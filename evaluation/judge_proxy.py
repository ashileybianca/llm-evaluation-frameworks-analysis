"""Proxy between the frameworks and Ollama: python judge_proxy.py

It forwards every request and adds up the tokens that Ollama reports for the judge calls. The evaluators read the
totals at GET /usage, before and after each metric. Each call is also appended to JUDGE_CALLS_PATH together with
the metric that the evaluator wrote in CURRENT_METRIC_PATH. It handles one call at a time on purpose.
"""

import json
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.append(str(Path(__file__).parents[1]))

from config import CURRENT_METRIC_PATH, JUDGE_CALLS_PATH, OLLAMA_URL, PROXY_URL

# Calls with a failed response or missing token counts have unknown token usage.
usage = {"input_tokens": 0, "output_tokens": 0, "n_calls_unknown_tokens": 0}


def request_params(body):
    """Sampling parameters sent by the framework, so every logged call shows how the judge was run."""
    try:
        data = json.loads(body)
    except ValueError:
        return {}
    options = data.get("options") or {}
    return {
        "model": data.get("model"),
        "temperature": data.get("temperature", options.get("temperature")),
        "top_p": data.get("top_p", options.get("top_p")),
        "max_tokens": data.get("max_tokens", options.get("num_predict")),
    }


class Proxy(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/usage":
            self.send_reply(200, json.dumps(usage).encode(), "application/json")
        else:
            self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        response = httpx.request(
            self.command, OLLAMA_URL + self.path, content=body, headers={"Content-Type": "application/json"}, timeout=None
        )

        if self.command == "POST" and self.path in ["/api/chat", "/api/generate", "/v1/chat/completions"]:
            self.count_tokens(response, body)

        self.send_reply(response.status_code, response.content, response.headers.get("Content-Type", "application/json"))

    def count_tokens(self, response, body):
        input_tokens = None
        output_tokens = None

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                data = {}

            if "prompt_eval_count" in data:  # /api/chat and /api/generate
                input_tokens = data["prompt_eval_count"]
                output_tokens = data.get("eval_count")
            elif data.get("usage"):  # /v1/chat/completions
                input_tokens = data["usage"].get("prompt_tokens")
                output_tokens = data["usage"].get("completion_tokens")

        if response.status_code != 200 or input_tokens is None or output_tokens is None:
            usage["n_calls_unknown_tokens"] += 1
        else:
            usage["input_tokens"] += input_tokens
            usage["output_tokens"] += output_tokens

        call = {
            "time": datetime.now(UTC).isoformat(timespec="microseconds"),
            "path": self.path,
            "status": response.status_code,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **request_params(body),
            **json.loads(CURRENT_METRIC_PATH.read_text(encoding="utf-8")),
        }
        JUDGE_CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JUDGE_CALLS_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(call, ensure_ascii=False) + "\n")

    def send_reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    address = urlparse(PROXY_URL)
    print(f"Judge proxy on {PROXY_URL} -> {OLLAMA_URL}. Stop with Ctrl+C.")
    HTTPServer((address.hostname, address.port), Proxy).serve_forever()
