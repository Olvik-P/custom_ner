"""Общие структуры данных для детекции PII в PrivacyGuard Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class PIISpan:
    """Обнаруженный спан PII в тексте.

    Attributes:
        start: Индекс начального символа.
        end: Индекс конечного символа.
        text: Исходный текст спана.
        entity_type: Тип PII-сущности (PHONE, EMAIL, PASSPORT и т.д.).
        source: Какой слой обнаружил ('pattern', 'natasha', 'context').
        confidence: Оценка уверенности (от 0.0 до 1.0).
    """

    start: int
    end: int
    text: str
    entity_type: str
    source: str = 'pattern'
    confidence: float = 1.0


@dataclass
class DetectionResult:
    """Результат прогона всех трёх слоёв детекции.

    Attributes:
        spans: Все обнаруженные PII-спаны, дедуплицированные и
            провалидированные.
        layer_stats: Количество обнаружений по каждому слою.
    """

    spans: list[PIISpan] = field(default_factory=list)
    layer_stats: dict[str, int] = field(
        default_factory=lambda: {
            'pattern': 0,
            'natasha': 0,
            'context': 0,
        }
    )


PreferSpan = Callable[[PIISpan, PIISpan], PIISpan]


def prefer_greater_end(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Тай-брейк по умолчанию: оставляет спан, который дотягивается дальше.

    Соответствует историческому правилу
    ``PatternMatcher.merge_overlapping`` — при точном равенстве (``end``
    совпадает) побеждает уже оставленный спан.
    """
    return incoming if incoming.end > kept.end else kept


def prefer_first(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Тай-брейк, всегда оставляющий уже оставленный (более ранний) спан."""
    return kept


def merge_overlapping_spans(
    spans: list[PIISpan],
    text: str,
    prefer: PreferSpan = prefer_greater_end,
) -> list[PIISpan]:
    """Сливает пересекающиеся спаны в их объединение, не теряя покрытия.

    Сортирует по ``(start, -end)`` и, если очередной спан начинается
    раньше, чем заканчивается уже оставленный, расширяет границы
    оставленного спана до объединения обоих диапазонов — в отличие от
    простого слияния "оставить победителя", это никогда не отбрасывает
    непересекающуюся часть проигравшего спана. ``text`` используется,
    чтобы пересчитать подстроку ``.text`` слитого спана для его
    (возможно, расширенных) границ. ``prefer(kept, incoming)`` решает,
    чьи ``entity_type``/``source``/``confidence`` достаются слитому
    спану, если они различаются.

    Args:
        spans: Спаны для слияния, возможно пересекающиеся.
        text: Полный исходный текст, в котором были обнаружены спаны.
        prefer: Функция тай-брейка; по умолчанию оставляет спан,
            который дотягивается дальше (по историческому правилу
            слоя pattern).

    Returns:
        Дедуплицированный список спанов, покрывающий полное объединение
        входных диапазонов, отсортированный по позиции начала.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: (s.start, -s.end))
    merged: list[PIISpan] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start >= last.end:
            merged.append(span)
            continue

        winner = prefer(last, span)
        new_start = min(last.start, span.start)
        new_end = max(last.end, span.end)
        merged[-1] = PIISpan(
            start=new_start,
            end=new_end,
            text=text[new_start:new_end],
            entity_type=winner.entity_type,
            source=winner.source,
            confidence=winner.confidence,
        )
    return merged


# Whitelist для распространённых слов, которые можно принять за имена
WHITELIST: set[str] = {
    # Дни недели
    'понедельник',
    'вторник',
    'среда',
    'четверг',
    'пятница',
    'суббота',
    'воскресенье',
    # Месяцы
    'январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь',
    # Распространённые слова, похожие на имена
    'роза',
    'лилия',
    'ромашка',
    'гвоздика',
    'фиалка',
    'ландыш',
    'вишня',
    'груша',
    'слива',
    'яблоко',
    'надежда',
    'вера',
    'любовь',
    'софия',
    'рай',
    'ад',
    'мир',
    'воля',
    'слава',
    'искра',
    'заря',
    'утро',
    'вечер',
    'ночь',
    'камень',
    'река',
    'озеро',
    'море',
    'поле',
    'лес',
    'зима',
    'весна',
    'лето',
    'осень',
    'север',
    'юг',
    'запад',
    'восток',
    'белый',
    'черный',
    'красный',
    'синий',
    'зеленый',
    'белое',
    'черное',
    'красное',
    'синее',
    'зеленое',
    'белая',
    'черная',
    'красная',
    'синяя',
    'зеленая',
}
