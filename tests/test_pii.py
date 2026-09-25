import pytest

from app.checks.pii import PIIChecker


@pytest.fixture(scope="module")
def checker():
    # Loads en_core_web_lg once for the whole test module -- mirrors the
    # "load once at startup" rule the checker itself is built around.
    return PIIChecker()


async def test_email_is_detected_and_redacted(checker):
    result = await checker.check("Contact me at jane.doe@example.com please.")
    assert result.found_any
    assert "jane.doe@example.com" not in result.redacted_text
    assert any(f.entity_type == "EMAIL_ADDRESS" for f in result.findings)


async def test_person_name_is_detected_and_redacted(checker):
    result = await checker.check("My name is Arjun Mehta and I need help.")
    assert result.found_any
    assert "Arjun Mehta" not in result.redacted_text
    assert any(f.entity_type == "PERSON" for f in result.findings)


async def test_clean_text_has_no_findings(checker):
    result = await checker.check("The sky was clear and the traffic was light.")
    assert not result.found_any
    assert result.redacted_text == "The sky was clear and the traffic was light."


async def test_date_time_false_positive_is_excluded(checker):
    # Regression test for a real finding, not a hypothetical: Presidio's
    # stock recognizer set flags the bare word "quarterly" as DATE_TIME at
    # score 0.85. ALLOWED_ENTITY_TYPES in pii.py excludes this category by
    # simply never asking the analyzer to check for it -- confirm it
    # actually stays clean now, not just documented as a known gap.
    result = await checker.check("Our quarterly numbers looked strong.")
    assert not result.found_any
    assert result.redacted_text == "Our quarterly numbers looked strong."


async def test_location_false_positive_is_excluded(checker):
    # Regression test for the demo-breaking case: this exact prompt got
    # "France" and "Paris" redacted out of an ordinary geography answer
    # before the ALLOWED_ENTITY_TYPES allowlist existed. A place name
    # isn't PII on its own -- confirm it no longer gets touched.
    result = await checker.check("The capital of France is Paris.")
    assert not result.found_any
    assert result.redacted_text == "The capital of France is Paris."
    assert not any(f.entity_type == "LOCATION" for f in result.findings)


async def test_in_pan_false_positive_is_excluded(checker):
    # Regression test for the third false-positive found while demoing
    # this live: Presidio's India PAN recognizer (IN_PAN, on by default
    # regardless of text locale) flagged the plain English words
    # "contacting" and "additional" inside a support-ticket template.
    # This is what moved pii.py from a denylist to an allowlist -- confirm
    # ordinary English support-template language stays untouched.
    text = "Thank you for contacting us. We may need additional information."
    result = await checker.check(text)
    assert not result.found_any
    assert result.redacted_text == text


async def test_only_allowlisted_entity_types_are_ever_returned(checker):
    # A broader guard than the individual false-positive regressions above:
    # whatever Presidio's default registry contains, nothing outside
    # PERSON/EMAIL_ADDRESS/PHONE_NUMBER should ever reach `findings`.
    from app.checks.pii import ALLOWED_ENTITY_TYPES

    text = (
        "My name is Arjun Mehta, email arjun@example.com, phone 555-123-4567, "
        "based in Paris, meeting scheduled for next Tuesday. PAN ABCDE1234F."
    )
    result = await checker.check(text)
    assert result.found_any
    assert all(f.entity_type in ALLOWED_ENTITY_TYPES for f in result.findings)


async def test_fires_identically_regardless_of_which_provider_produced_the_text(checker):
    # PS-3.3 / Phase 7's success criterion: PII redaction must fire the
    # same way no matter which adapter generated the raw text -- the
    # checker has no concept of "provider", so this is really asserting
    # that fact rather than testing provider-specific behavior.
    groq_style_output = "Sure, email me at test@groq-demo.com and I'll follow up."
    openai_style_output = "Sure, email me at test@groq-demo.com and I'll follow up."

    result_a = await checker.check(groq_style_output)
    result_b = await checker.check(openai_style_output)

    assert result_a.found_any == result_b.found_any == True
    assert result_a.redacted_text == result_b.redacted_text
