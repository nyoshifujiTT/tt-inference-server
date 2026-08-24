# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Auth helper for the diarization API tests.

``security.api_key_checker`` reads ``NO_AUTH`` once, at its own import time. A
test module that sets ``os.environ["NO_AUTH"]`` at the top therefore only wins
when it happens to be imported first; in a full-suite run another module can
import the checker earlier and the flag is ignored, so every request 401s.

Sending a real bearer token is order-independent, so tests use these headers
instead of relying on ``NO_AUTH``.
"""


def auth_headers():
    """Authorization header accepted whether or not NO_AUTH took effect."""
    from security.api_key_checker import API_KEY

    return {"Authorization": f"Bearer {API_KEY}"}
