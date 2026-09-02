"""Custom exceptions for PrivacyGuard Pipeline."""

from __future__ import annotations


class PrivacyGuardError(Exception):
    """Base exception for all PrivacyGuard pipeline errors."""


class LLMConnectionError(PrivacyGuardError):
    """Raised when LLM API is unreachable or returns an error."""


class LLMAuthenticationError(PrivacyGuardError):
    """Raised when LLM API key is invalid or missing."""


class NatashaModelError(PrivacyGuardError):
    """Raised when Natasha models fail to load or download."""


class TextTooLongError(PrivacyGuardError):
    """Raised when input text exceeds maximum allowed length."""


class ConfigurationError(PrivacyGuardError):
    """Raised when required configuration is missing or invalid."""


class PDFDependencyError(PrivacyGuardError):
    """Raised when a PDF-anonymization dependency is missing.

    Covers both the optional ``pdf`` package extra (PyMuPDF, pytesseract,
    Pillow) and the system Tesseract binary/language pack required for the
    OCR fallback.
    """
