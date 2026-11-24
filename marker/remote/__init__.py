"""Torch-free model backend: layout + OCR via a remote Chandra (OpenAI-compatible) server.

Everything fork-specific lives in this package so upstream merges stay cheap.
Enable with ``MODEL_BACKEND=chandra`` (see ``marker.settings``).
"""
