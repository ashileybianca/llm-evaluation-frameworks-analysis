"""Turns off the telemetry of the three frameworks. It must run before they are imported, and importing any
evaluator runs this file first."""

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["OPIK_TRACK_DISABLE"] = "true"
os.environ["OPIK_ANALYTICS_ENABLE"] = "false"
