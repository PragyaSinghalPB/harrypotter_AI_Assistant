"""Vercel serverless entry point: exposes the Flask app from starty.py."""

import os
import sys

# Make project-root modules (starty, process, prompt, rag, agent_service) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starty import app  # noqa: E402,F401
