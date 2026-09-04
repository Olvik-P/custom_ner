"""End-to-end PIIDetector + Masker tests for structured address spans.

Confirms the AddrExtractor-fix spans survive ContextualValidator's
merge/dedup pass and that each address component ends up masked as its
own token, not swallowed or left as plaintext.
"""

from __future__ import annotations

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.masker import Masker


class TestAddressComponentsSurviveDetectionAndMasking:
    def test_each_address_component_becomes_its_own_masked_token(
        self,
        detector: PIIDetector,
    ) -> None:
        text = (
            '620004, г. Екатеринбург, ул. Малышева, 101, '
            'тел. (343) 312-00-32, эл. почта: nadzor@egov66.ru'
        )
        result = detector.detect(text)
        masker = Masker()
        masked = masker.mask(text, result.spans)

        for original in (
            '620004',
            'Екатеринбург',
            'Малышева',
            '312-00-32',
            'nadzor@egov66.ru',
        ):
            assert original not in masked

        loc_tokens = [
            entry
            for entry in masker.mapping.values()
            if entry.entity_type == 'LOC'
        ]
        # index, city, street, house - each its own token, not one
        # merged address span.
        assert len(loc_tokens) >= 4

        demasked = masker.demask(masked)
        assert demasked == text
