"""Пользовательские исключения для PrivacyGuard Pipeline."""

from __future__ import annotations


class PrivacyGuardError(Exception):
    """Базовое исключение для всех ошибок PrivacyGuard pipeline."""


class LLMConnectionError(PrivacyGuardError):
    """Возникает, когда API LLM недоступен или возвращает ошибку."""


class LLMAuthenticationError(PrivacyGuardError):
    """Возникает, когда ключ API LLM невалиден или отсутствует."""


class NatashaModelError(PrivacyGuardError):
    """Возникает при сбое загрузки или скачивания моделей Natasha."""


class TextTooLongError(PrivacyGuardError):
    """Возникает, когда входной текст превышает максимальную длину."""


class ConfigurationError(PrivacyGuardError):
    """Возникает, когда обязательная конфигурация отсутствует или невалидна."""


class PDFDependencyError(PrivacyGuardError):
    """Возникает при отсутствии зависимости, нужной для анонимизации PDF.

    Покрывает и опциональную экстру пакета ``pdf`` (PyMuPDF,
    pytesseract, Pillow), и системный бинарник/языковой пакет
    Tesseract, нужные для OCR-фолбэка.
    """
