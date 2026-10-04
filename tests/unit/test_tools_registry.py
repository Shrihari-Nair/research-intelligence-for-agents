

def test_tool_mode_can_be_chosen_with_an_environment_variable(monkeypatch):
    import pytest

    from research_os.tools import build_registry
    monkeypatch.setenv("RESEARCH_OS_TOOLS", "bogus")
    with pytest.raises(ValueError, match="bogus"):
        build_registry()
    monkeypatch.setenv("RESEARCH_OS_TOOLS", "fixture")
    assert build_registry() is not None
