"""Check that the remote wait helper cancels abandoned GPU calls."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from modal_smoke import _call_async


def fake_function(get_result):
    call = SimpleNamespace(
        object_id="fc-test",
        get=SimpleNamespace(aio=get_result),
        cancel=SimpleNamespace(aio=AsyncMock()),
    )
    function = SimpleNamespace(spawn=SimpleNamespace(aio=AsyncMock(return_value=call)))
    return function, call


class RemoteWaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_call_returns_without_cancellation(self):
        function, call = fake_function(AsyncMock(return_value={"passed": True}))
        result = await _call_async(function, timeout=1)
        self.assertEqual(result, {"passed": True})
        call.cancel.aio.assert_not_awaited()

    async def test_deadline_cancels_pending_call(self):
        async def pending():
            await asyncio.Event().wait()

        function, call = fake_function(pending)
        with self.assertRaises(TimeoutError):
            await _call_async(function, timeout=0.02)
        call.cancel.aio.assert_awaited_once_with(terminate_containers=True)

    async def test_interruption_cancels_pending_call(self):
        started = asyncio.Event()

        async def pending():
            started.set()
            await asyncio.Event().wait()

        function, call = fake_function(pending)
        task = asyncio.create_task(_call_async(function, timeout=10))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        call.cancel.aio.assert_awaited_once_with(terminate_containers=True)
