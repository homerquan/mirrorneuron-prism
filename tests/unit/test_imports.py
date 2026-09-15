def test_import_package():
    import sys
    sys.path.insert(0, '/Users/homer/Sandbox/mirrorneuron-prism/src')
    import litellm_multicall
    assert litellm_multicall is not None
