"""Unit tests for the SQLite persistence layer, independent of FastAPI."""

from backend_app.chat_store import get_chat, init_db, list_chats, save_chat

SUMMARY_A = {
    "age": 34,
    "sex": "female",
    "chief_complaint": "Fever since yesterday",
    "symptoms": [{"name": "fever", "onset": "yesterday", "severity": "mild"}],
    "summary": "x",
    "risk_level": "LOW",
    "recommendation": "x",
    "requires_human": False,
    "missing_information": [],
    "specialty": "Primary Care",
    "scheduled_appointment": None,
}


def test_get_chat_returns_none_for_unknown_session(tmp_path):
    db_path = tmp_path / "chats.db"
    init_db(db_path)

    assert get_chat(db_path, "does-not-exist") is None


def test_save_then_get_round_trips_the_full_summary(tmp_path):
    db_path = tmp_path / "chats.db"
    init_db(db_path)

    save_chat(db_path, "session-a", SUMMARY_A)

    assert get_chat(db_path, "session-a") == SUMMARY_A


def test_list_chats_returns_lightweight_view_newest_first(tmp_path):
    db_path = tmp_path / "chats.db"
    init_db(db_path)

    save_chat(db_path, "session-a", SUMMARY_A)
    save_chat(db_path, "session-b", {**SUMMARY_A, "risk_level": "HIGH", "requires_human": True})

    listing = list_chats(db_path)

    assert [row["session_id"] for row in listing] == ["session-b", "session-a"]
    assert listing[0]["risk_level"] == "HIGH"
    assert listing[0]["requires_human"] is True
    # Lightweight view — not the full blob.
    assert "final_summary_json" not in listing[0]
    assert "symptoms" not in listing[0]


def test_save_chat_with_same_session_id_overwrites_not_duplicates(tmp_path):
    db_path = tmp_path / "chats.db"
    init_db(db_path)

    save_chat(db_path, "session-a", SUMMARY_A)
    save_chat(db_path, "session-a", {**SUMMARY_A, "risk_level": "HIGH"})

    assert len(list_chats(db_path)) == 1
    assert get_chat(db_path, "session-a")["risk_level"] == "HIGH"


def test_init_db_is_safe_to_call_repeatedly(tmp_path):
    db_path = tmp_path / "chats.db"
    init_db(db_path)
    init_db(db_path)  # must not raise

    save_chat(db_path, "session-a", SUMMARY_A)
    assert get_chat(db_path, "session-a") == SUMMARY_A
