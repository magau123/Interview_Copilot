from __future__ import annotations

from interview_copilot.storage.database import Database


def test_document_search_and_history(tmp_path) -> None:
    database = Database(tmp_path / "test.db")
    document_id = database.upsert_document("resume.md", "resume.md", "hash")
    database.replace_chunks(document_id, ["Python APIs", "Sales"], [[1.0, 0.0], [0.0, 1.0]])

    results = database.search([0.9, 0.1], limit=1)
    assert results[0].text == "Python APIs"

    session_id = database.start_session()
    turn_id = database.add_turn(session_id, "interviewer", "Tell me about Python")
    database.update_turn_translation(turn_id, "介绍一下 Python")
    assert "介绍一下 Python" in database.recent_context(session_id)
    database.end_session(session_id, "summary")
    database.close()


def test_reimport_replaces_chunks(tmp_path) -> None:
    database = Database(tmp_path / "test.db")
    document_id = database.upsert_document("resume.md", "resume.md", "one")
    database.replace_chunks(document_id, ["old"], [[1.0]])
    same_id = database.upsert_document("resume.md", "resume.md", "two")
    database.replace_chunks(same_id, ["new"], [[1.0]])
    assert document_id == same_id
    assert database.list_documents()[0]["chunk_count"] == 1
    database.close()
