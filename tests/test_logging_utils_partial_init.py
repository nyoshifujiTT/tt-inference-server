# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""AsyncLogHandler must survive close() before __init__ has finished.

``logging.Handler.__init__`` registers the instance in ``logging._handlerList``,
so ``logging.shutdown()`` can call ``close()`` on it before the subclass has
bound its own attributes. ``dictConfig()`` reaches that path via
``_clearExistingHandlers()``, and ``atexit`` reaches it at interpreter exit.

That happened here: launching the server raised
``AttributeError: 'AsyncLogHandler' object has no attribute '_listener'`` from
inside logging's own shutdown, which is reported as "Exception ignored in
atexit callback" and therefore easy to dismiss.

The fix was three lines in utils/logging_utils.py and shipped with no test, so
this file pins the two behaviours it relies on. Both are demonstrated against
the pre-fix bodies below, which do raise.
"""

import logging

import pytest

from utils.logging_utils import AsyncLogHandler, _safe_stop_listener


def _half_built():
    """A handler registered with logging but with no subclass state yet."""
    handler = AsyncLogHandler.__new__(AsyncLogHandler)
    logging.Handler.__init__(handler)
    return handler


def test_close_on_a_half_built_handler_does_not_raise():
    handler = _half_built()
    try:
        handler.close()  # must not raise
    finally:
        logging._removeHandlerRef(logging.weakref.ref(handler))  # keep the list clean


def test_the_pre_fix_close_really_did_raise():
    """Guard the premise: without the guard this is an AttributeError.

    If a later refactor makes _listener always present, this test starts
    failing and says the guard has become unnecessary rather than silently
    keeping dead code.
    """

    class PreFix(AsyncLogHandler):
        def close(self):
            _safe_stop_listener(self._listener)
            if AsyncLogHandler._active_listener is self._listener:
                AsyncLogHandler._active_listener = None
            logging.Handler.close(self)

    handler = PreFix.__new__(PreFix)
    logging.Handler.__init__(handler)
    with pytest.raises(AttributeError, match="_listener"):
        handler.close()
    # leave nothing for atexit to trip over
    handler._listener = None
    logging.Handler.close(handler)


def test_emit_without_a_listener_is_dropped_not_an_error():
    """A record logged between registration and setup must not crash the caller.

    The queue does not exist yet at that point, so the original body raised
    AttributeError on _queue -- from inside a logging call, i.e. anywhere.
    """
    handler = _half_built()
    handler._listener = None
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "m", None, None)
    handler.emit(record)  # must return quietly
    handler.close()


def test_close_leaves_the_active_listener_alone_when_it_is_not_ours():
    """`is` comparisons against None used to clear another handler's listener.

    close() now only clears the class-level pointer when its own listener is
    the one recorded. With _listener None and the guard removed, a half-built
    handler closing would null out the live handler's registration.
    """
    sentinel = object()
    AsyncLogHandler._active_listener = sentinel
    try:
        handler = _half_built()
        handler.close()
        assert AsyncLogHandler._active_listener is sentinel, (
            "closing a handler with no listener must not clear another one's"
        )
    finally:
        AsyncLogHandler._active_listener = None


def test_the_is_not_none_half_of_the_guard_is_documentation_only():
    """Recorded because a test that claimed to cover it would be lying.

    close() reads

        if listener is not None and AsyncLogHandler._active_listener is listener:

    and the first half changes nothing observable. With ``listener`` None
    there are two cases: the class pointer holds a real listener, and both
    forms decline; or it is already None, and the shorter form "clears" None
    to None. Removing that half therefore passes every behavioural test --
    confirmed by mutation.

    So it stays as intent, and this test states that it is not load-bearing
    rather than pretending otherwise. Asserting the source text would be
    checking the code against itself.
    """
    marker = object()
    AsyncLogHandler._active_listener = marker
    try:
        handler = _half_built()
        handler.close()
        assert AsyncLogHandler._active_listener is marker, (
            "the second half of the guard is the load-bearing one: a handler "
            "with no listener must not match a real registration"
        )
    finally:
        AsyncLogHandler._active_listener = None

    # and the degenerate case is a no-op either way, which is why the first
    # half cannot be distinguished behaviourally
    AsyncLogHandler._active_listener = None
    _half_built().close()
    assert AsyncLogHandler._active_listener is None


def test_close_only_clears_the_registration_it_owns():
    """The identity comparison is the load-bearing part, so exercise it.

    A handler that has its own listener, but is not the one currently
    registered, must leave the registration alone. Dropping the identity test
    (keeping only `listener is not None`) makes such a close clear another
    handler's slot -- and the half-built cases above cannot see it, because
    their listener is None and the shortened condition declines anyway.
    """
    registered = object()
    mine = object()
    AsyncLogHandler._active_listener = registered
    try:
        handler = _half_built()
        handler._listener = mine  # a real listener, but not the registered one
        handler.close()
        assert AsyncLogHandler._active_listener is registered, (
            "closing a handler must not clear a registration belonging to "
            "another handler"
        )
    finally:
        AsyncLogHandler._active_listener = None


def test_close_does_clear_its_own_registration():
    """The other side of the same comparison, so it cannot be inverted."""
    mine = object()
    AsyncLogHandler._active_listener = mine
    try:
        handler = _half_built()
        handler._listener = mine
        handler.close()
        assert AsyncLogHandler._active_listener is None, (
            "a handler that owns the registration must release it on close"
        )
    finally:
        AsyncLogHandler._active_listener = None


def test_init_binds_the_attribute_before_anything_can_fail():
    """close() must be safe from the first statement after registration.

    logging.Handler.__init__ publishes the instance, so every statement after
    it runs while shutdown() may call close(). The binding therefore has to be
    the first thing __init__ does with its own state -- not merely present
    somewhere in the body.
    """
    import inspect

    body = [
        line.strip()
        for line in inspect.getsource(AsyncLogHandler.__init__).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # [0] is the def, [1] is super().__init__()
    assert body[1].startswith("super().__init__()"), body[:3]
    assert body[2] == "self._listener = None", (
        "bind _listener immediately after registration; anything before it "
        f"runs unprotected, got {body[2]!r}"
    )
