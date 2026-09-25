import pytest
from pydantic import ValidationError

from app.policy import CheckAction, Policy


def test_from_yaml_parses_default_policy(tmp_path):
    yaml_content = """
checks:
  pii:
    action: redact
  topic_denial:
    action: block
    topics: [medical_advice, legal_advice]
    threshold: 0.75
"""
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml_content)

    policy = Policy.from_yaml(policy_file)

    assert policy.checks.pii.action == CheckAction.REDACT
    assert policy.checks.topic_denial.action == CheckAction.BLOCK
    assert policy.checks.topic_denial.topics == ["medical_advice", "legal_advice"]
    assert policy.checks.topic_denial.threshold == 0.75


def test_missing_sections_fall_back_to_defaults(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("checks: {}\n")

    policy = Policy.from_yaml(policy_file)

    assert policy.checks.pii.action == CheckAction.REDACT
    assert policy.checks.topic_denial.topics == []


def test_threshold_out_of_range_rejected(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        """
checks:
  topic_denial:
    threshold: 1.5
"""
    )
    with pytest.raises(ValidationError):
        Policy.from_yaml(policy_file)


def test_unknown_action_rejected(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        """
checks:
  pii:
    action: incinerate
"""
    )
    with pytest.raises(ValidationError):
        Policy.from_yaml(policy_file)
