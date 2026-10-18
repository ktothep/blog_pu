import re

from api.path.tailor import calculate_ats_score, extract_keyword_hits


def test_extract_keyword_hits_finds_case_insensitive_matches():
    job_description = "We are hiring a Python engineer with AWS and Kubernetes experience."
    resume_text = "Experienced Python developer with AWS and Docker."

    hits = extract_keyword_hits(job_description, resume_text)

    assert "python" in hits
    assert "aws" in hits
    assert "kubernetes" not in hits


def test_calculate_ats_score_returns_percentage_for_keyword_coverage():
    job_description = "Python AWS Kubernetes Docker FastAPI SQL React"
    resume_text = "Python AWS Docker FastAPI SQL"

    score = calculate_ats_score(job_description, resume_text)

    assert 60 <= score <= 80
    assert score > 0
