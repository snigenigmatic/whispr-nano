"""Classify reference tokens by writing system."""

from __future__ import annotations

import unicodedata


def token_class(token: str) -> str:
    latin = any("LATIN" in unicodedata.name(c, "") for c in token)
    devanagari = any("DEVANAGARI" in unicodedata.name(c, "") for c in token)
    if latin and devanagari:
        return "MIX"
    if latin:
        return "LAT"
    if devanagari:
        return "DEV"
    if token and all(c.isdigit() for c in token):
        return "NUM"
    return "OTHER"
