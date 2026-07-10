from __future__ import annotations

import pytest

from bot.journal import Journal


@pytest.fixture()
def journal(tmp_path):
    j = Journal(tmp_path / "test.sqlite")
    yield j
    j.close()
