NS_TO = {"ns": 1, "ms": 1e6, "s": 1e9}

def check_monotonic_rows(rows: np.ndarray) -> None:
    last_ts = 0
    for row in rows:
        ts = row[0]
        assert last_ts < ts
        last_ts = ts
