"""Tests for the frozen calibration / holdout split.

The split's whole value is that it was fixed before any gate result was seen.
So the properties worth testing are the ones that would let it silently stop
being a holdout: drift between runs, overlap between the two sides, questions
going missing, or a holdout that is accidentally all Easy questions from one
document.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.eval_loader import load_all
from src.evaluation.splits import (
    DEFAULT_SPLIT_PATH,
    Split,
    build_split,
    freeze,
    load_split,
)


@pytest.fixture(scope="module")
def questions():
    return load_all()


@pytest.fixture(scope="module")
def split():
    return load_split()


def test_frozen_split_file_exists():
    """If this fails the file was deleted — regenerating it after seeing gate
    results would silently destroy the holdout."""
    assert DEFAULT_SPLIT_PATH.exists(), (
        "evaluation_v2/split_v1.json is missing. It must be committed."
    )


def test_split_is_deterministic(questions):
    """Same inputs, same split — regardless of process or row order."""
    a, u, m = questions
    first = build_split(a, u, m)
    second = build_split(list(reversed(a)), list(reversed(u)), list(reversed(m)))
    assert first.to_dict() == second.to_dict()


def test_frozen_file_matches_what_the_code_computes(questions):
    """Guards against the file and the algorithm drifting apart."""
    a, u, m = questions
    assert build_split(a, u, m).to_dict() == load_split().to_dict()


def test_sides_are_disjoint(split):
    assert not (split.calibration_ids() & split.holdout_ids())


def test_every_question_lands_on_exactly_one_side(questions, split):
    a, u, m = questions
    all_ids = {q.question_id for q in a} | {q.question_id for q in u} | {q.question_id for q in m}
    assert split.calibration_ids() | split.holdout_ids() == all_ids
    assert len(all_ids) == 57


def test_all_three_classes_are_represented_in_the_holdout(split):
    """A holdout with no unanswerable questions cannot test abstention, which
    is half the confidence-gating claim."""
    assert split.answerable_holdout
    assert split.unanswerable_holdout
    assert split.ambiguous_holdout


def test_holdout_is_large_enough_to_mean_something(split):
    """Below ~15 held out, one question moves the reported rate more than the
    effect being measured."""
    assert len(split.holdout_ids()) >= 15


def test_holdout_difficulty_mix_resembles_the_full_set(questions, split):
    """With 44 questions an unstratified draw can easily be all-Easy, and the
    reported number would then describe the draw rather than the gate."""
    a, _, _ = questions
    by_id = {q.question_id: q for q in a}
    holdout_difficulties = {by_id[i].gold.difficulty for i in split.answerable_holdout}
    all_difficulties = {q.gold.difficulty for q in a}
    assert holdout_difficulties == all_difficulties


def test_holdout_spans_multiple_documents(questions, split):
    a, _, _ = questions
    by_id = {q.question_id: q for q in a}
    docs = {by_id[i].gold.expected_document_id for i in split.answerable_holdout}
    assert len(docs) >= 5


def test_ids_for_rejects_unknown_side(split):
    with pytest.raises(ValueError):
        split.ids_for("training")


def test_ids_for_all_means_no_filter(split):
    assert split.ids_for("all") is None


def test_freeze_refuses_to_overwrite(tmp_path):
    """Regenerating a split after results are known invalidates it, so this
    must be a deliberate act, not an accident."""
    p = tmp_path / "split.json"
    p.write_text("{}")
    with pytest.raises(FileExistsError):
        freeze(p)


def test_load_split_fails_loudly_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path / "nope.json")


def test_round_trips_through_json(split):
    assert Split.from_dict(json.loads(json.dumps(split.to_dict()))).to_dict() == split.to_dict()
