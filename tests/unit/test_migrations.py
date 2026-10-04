from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from research_os.memory.models import Base


def cfg(url):
    c = Config("alembic.ini")
    c.set_main_option("script_location", "migrations")
    c.cmd_opts = type("O", (), {"x": [f"url={url}"]})()
    return c


def test_upgrade_head_creates_exactly_the_model_schema_and_downgrade_removes_it(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    command.upgrade(cfg(url), "head")
    insp = inspect(create_engine(url))
    tables = set(insp.get_table_names()) - {"alembic_version"}
    assert tables == set(Base.metadata.tables)
    for name, table in Base.metadata.tables.items():
        assert {c["name"] for c in insp.get_columns(name)} == {c.name for c in table.columns}, name
    command.downgrade(cfg(url), "base")
    assert set(inspect(create_engine(url)).get_table_names()) - {"alembic_version"} == set()


def test_there_is_a_single_migration_head_matching_the_models():
    from alembic.script import ScriptDirectory
    heads = ScriptDirectory.from_config(cfg("sqlite://")).get_heads()
    assert heads == ["0001"]


def test_store_migrate_applies_alembic_on_a_fresh_sqlite_file(tmp_path):
    from research_os.memory.embeddings import HashEmbedder
    from research_os.memory.research_store import ResearchStore
    s = ResearchStore(f"sqlite:///{tmp_path / 'fresh' / 'r.db'}", embedder=HashEmbedder())
    s.migrate()
    s.create_run("a", "q about things", "mock")
    assert s.get_run("a")["status"] == "RUNNING"
