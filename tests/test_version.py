from plantcv_mcp import __version__, plantcv_version


def test_package_version_is_a_string():
    assert isinstance(__version__, str)
    assert __version__


def test_plantcv_version_is_pinned_4_11_3():
    # plantcv.__version__ does NOT exist; we must read package metadata.
    assert plantcv_version() == "4.11.3"
