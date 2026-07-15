"""
JobForge AI — Sponsor Register Matching Tests.

"JD mentions sponsorship" and "employer holds a Home Office sponsor licence"
are distinct signals (see connectors/sponsor_register.py). These tests cover
the matching logic against a small fixture register — no network calls.
"""

from __future__ import annotations

from jobforge.connectors.sponsor_register import (
    SponsorRegisterMatcher,
    normalize_company_name,
)


class TestNormalizeCompanyName:
    def test_strips_ltd_suffix(self):
        assert normalize_company_name("Acme Ltd") == normalize_company_name("ACME LIMITED")

    def test_strips_trading_as(self):
        assert normalize_company_name("Acme Ltd T/A Acme Foods") == normalize_company_name(
            "Acme Ltd Trading As Acme Foods"
        )

    def test_strips_punctuation_and_extra_whitespace(self):
        assert normalize_company_name("Acme, Inc.") == normalize_company_name("Acme   Inc")

    def test_empty_input_returns_empty_string(self):
        assert normalize_company_name("") == ""


class TestSponsorRegisterMatcher:
    def _write_register(self, tmp_path, rows: list[str]):
        csv_path = tmp_path / "register.csv"
        content = "Organisation Name,Town/City,County,Type & Rating,Route\n" + "\n".join(rows)
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_exact_match_after_normalization(self, tmp_path):
        csv_path = self._write_register(
            tmp_path, [" DeepMind Technologies Limited,London,,Worker (A rating),Skilled Worker"]
        )
        matcher = SponsorRegisterMatcher(csv_path)

        assert matcher.is_licensed_sponsor("DeepMind Technologies Ltd") is True

    def test_fuzzy_match_for_near_miss_spelling(self, tmp_path):
        csv_path = self._write_register(
            tmp_path, [" DeepMind Technologies Limited,London,,Worker (A rating),Skilled Worker"]
        )
        matcher = SponsorRegisterMatcher(csv_path)

        # Single-character typo ("Technologie" vs "Technologies") — too close to be
        # a coincidence, but not an exact normalised match either.
        assert matcher.is_licensed_sponsor("DeepMind Technologie Ltd") is True

    def test_unrelated_company_does_not_match(self, tmp_path):
        csv_path = self._write_register(
            tmp_path, [" DeepMind Technologies Limited,London,,Worker (A rating),Skilled Worker"]
        )
        matcher = SponsorRegisterMatcher(csv_path)

        assert matcher.is_licensed_sponsor("A Completely Different Company") is False

    def test_empty_company_name_does_not_match(self, tmp_path):
        csv_path = self._write_register(
            tmp_path, [" DeepMind Technologies Limited,London,,Worker (A rating),Skilled Worker"]
        )
        matcher = SponsorRegisterMatcher(csv_path)

        assert matcher.is_licensed_sponsor("") is False

    def test_len_reflects_loaded_entries(self, tmp_path):
        csv_path = self._write_register(
            tmp_path,
            [
                " Acme Ltd,London,,Worker (A rating),Skilled Worker",
                " Beta Corp,Manchester,,Worker (A rating),Skilled Worker",
            ],
        )
        matcher = SponsorRegisterMatcher(csv_path)

        assert len(matcher) == 2
