


from mz import helpers


def test_lru_dict():
    d = helpers.LRUDict(capacity=3)
    d[1] = "foo"
    d[2] = "bar"
    d[3] = "baz"
    # Fetch `1`, so `2` becomes least recently used.
    _ = d[1]
    d[4] = "bax"
    assert 2 not in d
    assert d[1] == "foo"
    assert d[3] == "baz"
    assert d[4] == "bax"