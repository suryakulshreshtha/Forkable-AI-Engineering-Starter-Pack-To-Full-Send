"""The judge for drill 7.3."""


class Flaky:
    """Fails `failures` times with `error`, then returns `value`."""

    def __init__(self, failures, value="ok", error=TimeoutError):
        self.failures, self.value, self.error, self.calls = failures, value, error, 0

    def __call__(self, *args):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error(f"failure {self.calls}")
        return self.value


def test_estimate_tokens(m):
    assert [m.estimate_tokens(t) for t in ["", "abcd", "abcde", "x" * 40]] == [0, 1, 2, 10]


def test_first_try_success_never_sleeps(m):
    slept = []
    assert m.retry(Flaky(0), sleep=slept.append) == "ok" and slept == []


def test_backoff_doubles_between_failures(m):
    slept, fn = [], Flaky(2)
    assert m.retry(fn, attempts=4, base_delay=0.5, sleep=slept.append) == "ok"
    assert slept == [0.5, 1.0] and fn.calls == 3


def test_gives_up_and_reraises_the_last_error(m):
    slept, fn = [], Flaky(10)
    try:
        m.retry(fn, attempts=3, base_delay=1, sleep=slept.append)
    except TimeoutError as exc:
        assert str(exc) == "failure 3", "re-raise the last error, not the first"
    else:
        raise AssertionError("after the last attempt the error must propagate")
    assert fn.calls == 3 and slept == [1, 2], "no sleep after the final failure"


def test_delays_are_capped(m):
    slept = []
    m.retry(Flaky(5), attempts=6, base_delay=1, max_delay=4, sleep=slept.append)
    assert slept == [1, 2, 4, 4, 4]


def test_non_retryable_errors_escape_immediately(m):
    slept, fn = [], Flaky(1, error=ValueError)
    try:
        m.retry(fn, attempts=5, sleep=slept.append, retry_on=(TimeoutError,))
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError is not in retry_on, so it should propagate")
    assert fn.calls == 1 and slept == []


def test_budget_charges_and_refuses(m):
    budget = m.TokenBudget(100)
    budget.charge(60)
    assert budget.used == 60 and budget.remaining == 40 and budget.can_afford(40) and not budget.can_afford(41)
    try:
        budget.charge(50)
    except m.BudgetExceeded:
        pass
    else:
        raise AssertionError("charging past the limit must raise BudgetExceeded")
    assert budget.used == 60, "a rejected charge must not be recorded"


def test_negative_charges_are_rejected(m):
    try:
        m.TokenBudget(10).charge(-5)
    except ValueError:
        return
    raise AssertionError("charge(-5) should raise ValueError")


def test_every_retry_costs_tokens(m):
    budget, fn = m.TokenBudget(1000), Flaky(2, value="answer")
    prompt = "x" * 40   # 10 tokens per attempt
    assert m.guarded_call(fn, prompt, budget, sleep=lambda s: None) == "answer"
    assert budget.used == 30, "three attempts at 10 tokens each"


def test_the_budget_stops_runaway_retries(m):
    budget, fn = m.TokenBudget(25), Flaky(5)
    try:
        m.guarded_call(fn, "x" * 40, budget, attempts=5, sleep=lambda s: None)
    except m.BudgetExceeded:
        pass
    else:
        raise AssertionError("the third attempt would cost 30 > 25 and must raise BudgetExceeded")
    assert fn.calls == 2 and budget.used == 20
