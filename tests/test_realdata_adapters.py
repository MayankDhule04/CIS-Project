"""Tests for the real-data adapters (CI-safe: synthetic fixtures only).

These never touch the real 20 GB dataset or any real answers file. They build
tiny temp fixtures that mirror the CERT r4.2 formats and assert the adapters
produce the exact helper files the pipeline already consumes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ueba.realdata.adapters import (
    build_labels_from_answers,
    build_roles_from_ldap,
)


# ---------------------------------------------------------------------------
# LDAP -> roles.csv
# ---------------------------------------------------------------------------


def _write_ldap_month(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a minimal LDAP monthly snapshot with the real column header."""
    header = (
        "employee_name,user_id,email,role,business_unit,"
        "functional_unit,department,team,supervisor"
    )
    lines = [header]
    for user_id, role in rows:
        lines.append(
            f"Name {user_id},{user_id},{user_id}@dtaa.com,{role},1,fu,dept,team,sup"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_roles_uses_most_recent_month(tmp_path: Path) -> None:
    """A user whose role changes resolves to the most-recent month's role."""
    ldap = tmp_path / "LDAP"
    ldap.mkdir()
    # Older month: ABC0001 is a Salesman; XYZ0002 is an Engineer.
    _write_ldap_month(
        ldap / "2010-01.csv",
        [("ABC0001", "Salesman"), ("XYZ0002", "Engineer")],
    )
    # Newer month: ABC0001 changed to Manager; XYZ0002 unchanged.
    _write_ldap_month(
        ldap / "2010-02.csv",
        [("ABC0001", "Manager"), ("XYZ0002", "Engineer")],
    )

    out_path = tmp_path / "roles.csv"
    roles = build_roles_from_ldap(ldap, out_path=out_path)

    mapping = dict(zip(roles["user"].tolist(), roles["role"].tolist()))
    assert mapping["ABC0001"] == "Manager"  # most-recent wins
    assert mapping["XYZ0002"] == "Engineer"

    # roles.csv written with the exact columns the extractor reads.
    assert out_path.is_file()
    written = pd.read_csv(out_path, dtype=str)
    assert list(written.columns) == ["user", "role"]
    assert set(written["user"]) == {"ABC0001", "XYZ0002"}


def test_build_roles_missing_dir_raises(tmp_path: Path) -> None:
    """A missing LDAP directory raises an actionable FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        build_roles_from_ldap(tmp_path / "does-not-exist")


def test_build_roles_empty_dir_raises(tmp_path: Path) -> None:
    """An LDAP directory with no CSVs raises."""
    empty = tmp_path / "LDAP"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        build_roles_from_ldap(empty)


# ---------------------------------------------------------------------------
# answers -> labels.csv
# ---------------------------------------------------------------------------


def test_build_labels_expands_insiders_range(tmp_path: Path) -> None:
    """A published insiders.csv range expands to per-day positive user-days."""
    insiders = tmp_path / "insiders.csv"
    insiders.write_text(
        "dataset,scenario,details,user,start,end\n"
        "r4.2,2,some details,CDE1846,01/02/2011 08:00:00,01/04/2011 17:00:00\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "labels.csv"
    labels = build_labels_from_answers(insiders, out_path=out_path)

    # 01/02, 01/03, 01/04 -> three positive user-days, all insider=1.
    assert list(labels.columns) == ["user", "day", "insider"]
    assert set(labels["day"]) == {"2011-01-02", "2011-01-03", "2011-01-04"}
    assert (labels["user"] == "CDE1846").all()
    assert (labels["insider"] == 1).all()

    written = pd.read_csv(out_path)
    assert list(written.columns) == ["user", "day", "insider"]
    assert len(written) == 3


def test_build_labels_from_answers_directory(tmp_path: Path) -> None:
    """A headerless per-scenario answer file is mined for user + date."""
    answers = tmp_path / "answers" / "r4.2-2"
    answers.mkdir(parents=True)
    # CERT per-scenario rows mirror a raw log line: type,id,date,user,...
    (answers / "CDE1846.csv").write_text(
        "logon,{id-1},01/15/2011 02:17:34,CDE1846,PC-1,Logon\n"
        "device,{id-2},01/15/2011 02:20:00,CDE1846,PC-1,Connect\n"
        "http,{id-3},01/16/2011 23:05:10,CDE1846,PC-1,http://wikileaks.org/x\n",
        encoding="utf-8",
    )

    labels = build_labels_from_answers(tmp_path / "answers")

    assert set(labels["day"]) == {"2011-01-15", "2011-01-16"}
    assert (labels["user"] == "CDE1846").all()
    assert (labels["insider"] == 1).all()


def test_build_labels_event_days_restriction(tmp_path: Path) -> None:
    """event_days narrows an insider range to days with actual activity."""
    insiders = tmp_path / "insiders.csv"
    insiders.write_text(
        "dataset,scenario,details,user,start,end\n"
        "r4.2,2,d,CDE1846,01/02/2011,01/05/2011\n",
        encoding="utf-8",
    )
    # The user only has events on two of the four days in range.
    event_days = pd.DataFrame(
        {"user": ["CDE1846", "CDE1846"], "day": ["2011-01-03", "2011-01-05"]}
    )

    labels = build_labels_from_answers(insiders, event_days=event_days)

    assert set(labels["day"]) == {"2011-01-03", "2011-01-05"}


def test_build_labels_missing_path_raises(tmp_path: Path) -> None:
    """A missing answers path raises a clear error and fabricates nothing."""
    with pytest.raises(FileNotFoundError):
        build_labels_from_answers(tmp_path / "no-answers-here")


# ---------------------------------------------------------------------------
# answers (multi-release tree) -> r4.2-only labels
# ---------------------------------------------------------------------------


def _write_headerless_answer(path: Path, user: str, rows: list[tuple[str, str]]) -> None:
    """Write a CERT per-insider file: eventtype,{id},date,user,pc,extra."""
    lines = []
    for event_type, date in rows:
        lines.append(f"{event_type},{{id-x}},{date},{user},PC-1,detail")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_multi_release_answers(tmp_path: Path) -> Path:
    """Build a tiny answers/ tree mixing r4.2 scenario dirs and other releases."""
    answers = tmp_path / "answers"
    (answers / "r4.2-1").mkdir(parents=True)
    (answers / "r4.2-2").mkdir(parents=True)
    (answers / "r4.1-1").mkdir(parents=True)  # a different release as a dir
    # r4.2 scenario 1 insider AAM0658: two malicious days.
    _write_headerless_answer(
        answers / "r4.2-1" / "r4.2-1-AAM0658.csv",
        "AAM0658",
        [("logon", "10/23/2010 01:34:19"), ("device", "10/29/2010 00:06:47")],
    )
    # r4.2 scenario 2 insider CDE1846: one malicious day.
    _write_headerless_answer(
        answers / "r4.2-2" / "r4.2-2-CDE1846.csv",
        "CDE1846",
        [("http", "01/16/2011 23:05:10")],
    )
    # A non-r4.2 insider as a single CSV at the root (r5.1) — must be excluded.
    (answers / "r5.1-1.csv").write_text(
        "dataset,scenario,details,user,start,end\n"
        "5.1,1,d,JTC1885,12/03/2010 06:05:31,12/08/2010 23:35:42\n",
        encoding="utf-8",
    )
    # A non-r4.2 insider inside a different-release directory — must be excluded.
    _write_headerless_answer(
        answers / "r4.1-1" / "r4.1-1-ABB0427.csv",
        "ABB0427",
        [("logon", "08/21/2010 05:20:34")],
    )
    return answers


def test_collect_release_scenario_dirs_selects_only_r42(tmp_path: Path) -> None:
    """Only the r4.2-* scenario directories are returned, not other releases."""
    from src.ueba.realdata.adapters import collect_release_scenario_dirs

    answers = _make_multi_release_answers(tmp_path)
    dirs = collect_release_scenario_dirs(answers, "r4.2")

    names = sorted(d.name for d in dirs)
    assert names == ["r4.2-1", "r4.2-2"]  # r4.1-1 dir excluded


def test_build_labels_for_release_keeps_only_r42(tmp_path: Path) -> None:
    """r4.2 label building excludes every other release's users and days."""
    from src.ueba.realdata.adapters import build_labels_for_release

    answers = _make_multi_release_answers(tmp_path)
    out_path = tmp_path / "labels.csv"
    labels = build_labels_for_release(answers, release="r4.2", out_path=out_path)

    # Only the two r4.2 insiders, with their exact malicious days.
    assert set(labels["user"]) == {"AAM0658", "CDE1846"}
    assert set(labels["day"]) == {"2010-10-23", "2010-10-29", "2011-01-16"}
    assert (labels["insider"] == 1).all()
    # The non-r4.2 insiders never appear.
    assert "JTC1885" not in set(labels["user"])
    assert "ABB0427" not in set(labels["user"])

    written = pd.read_csv(out_path)
    assert list(written.columns) == ["user", "day", "insider"]


def test_build_labels_for_release_missing_release_raises(tmp_path: Path) -> None:
    """Requesting a release with no scenario directories raises, never empties."""
    from src.ueba.realdata.adapters import build_labels_for_release

    answers = _make_multi_release_answers(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_labels_for_release(answers, release="r6.9")


# ---------------------------------------------------------------------------
# End-to-end-on-synthetic: the pipeline consumers accept the adapter output.
# ---------------------------------------------------------------------------


def test_pipeline_consumers_accept_adapter_output(tmp_path: Path) -> None:
    """roles.csv/labels.csv from the adapters feed the existing consumers."""
    from src.ueba.ingestion.subset import read_insiders, read_population

    ldap = tmp_path / "LDAP"
    ldap.mkdir()
    _write_ldap_month(
        ldap / "2010-01.csv",
        [("ABC0001", "Salesman"), ("XYZ0002", "Engineer")],
    )
    build_roles_from_ldap(ldap, out_path=tmp_path / "roles.csv")

    insiders_file = tmp_path / "insiders.csv"
    insiders_file.write_text(
        "dataset,scenario,details,user,start,end\n"
        "r4.2,1,d,ABC0001,01/02/2011,01/02/2011\n",
        encoding="utf-8",
    )
    build_labels_from_answers(insiders_file, out_path=tmp_path / "labels.csv")

    # subset.read_population reads roles.csv; read_insiders reads labels.csv.
    assert read_population(tmp_path) == {"ABC0001", "XYZ0002"}
    assert read_insiders(tmp_path) == {"ABC0001"}
