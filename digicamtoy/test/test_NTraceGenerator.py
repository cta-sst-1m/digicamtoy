from digicamtoy.tracegenerator import NTraceGenerator


def test_n_events():

    test_n_events = [0, 1, 10, 100]
    for n_events in test_n_events:
        toy = NTraceGenerator(
            nsb_rate=0.000001,  # small rate for faster tests
            n_events=n_events
        )
        events = [e for e in toy]
        assert len(events) == n_events
