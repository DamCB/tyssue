"""
Event management module
=======================


"""

import logging
import random
from collections import deque
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class EventManager:
    """
    Behavior management class based on two deques, the current and next one.



    """

    def __init__(self, element=None, logfile=None):
        """Creates an events class

        Parameters
        ----------
        element : str
           element on which the events occur, e.g face or cell, optional
        logfile : str, default None
           if logfile is not None, will create a logging handler
           for this file where each event will be logged
        """
        self.current = deque()
        self.next = deque()
        self.element = element
        self.current.append((wait, {"face_id": -1, "n_steps": 1}))
        self.clock = 0
        if logfile is not None:
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.debug("# Started logging at %s", datetime.now().isoformat())
            logger.debug(f"time, {self.element} index, event")

    def extend(self, events):
        """
        Add a list of events to the next deque

        Parameters
        ----------
        events : list of tuples (behavior, [elem_id, args, kwargs])
            the three last elements don't need to be passed

        """
        for event in events:
            self.append(event[0], **event[1])

    def append(self, behavior, **kwargs):
        """Add an event to the manager's next deque

        behavior is a function whose signature is
        ..code :
            behavior(sheet, manager, **kwargs)

        this function itself might populate the managers next deque

        Parameters
        ----------
        behavior : function
        kwargs : dict defaults to {}
            keywords arguments to the behavior function
            if `"unique"` is True (the default), the event is skipped
            when the same behavior with the same kwargs is already queued.
        """
        self._append_unique(self.next, behavior, kwargs)

    def extend_current(self, events, front=False):
        """
        Add a list of events to the current deque

        Parameters
        ----------
        events : list of tuples (behavior, [elem_id, args, kwargs])
            the three last elements don't need to be passed
        front : bool, default False
            if True, the events are put ahead of those already in the
            current deque (keeping their own order), so they are executed
            first by the next `execute`.

        """
        if not front:
            for event in events:
                self.append_current(event[0], **event[1])
            return
        added = deque()
        for event in events:
            behavior, kwargs = event[0], event[1]
            if kwargs.get("unique", True) and (
                _is_queued(self.current, behavior, kwargs)
                or _is_queued(added, behavior, kwargs)
            ):
                continue
            added.append((behavior, kwargs))
        self.current.extendleft(reversed(added))

    def append_current(self, behavior, **kwargs):
        """Add an event to the manager's current deque

        Events added while `execute` is running are executed in the
        same step, after the events already in the current deque.

        Parameters
        ----------
        behavior : function
        kwargs : dict defaults to {}
            keywords arguments to the behavior function
            if `"unique"` is True (the default), the event is skipped
            when the same behavior with the same kwargs is already queued.
        """
        self._append_unique(self.current, behavior, kwargs)

    def _append_unique(self, queue, behavior, kwargs):
        """Append (behavior, kwargs) to queue unless an identical event is there."""
        if kwargs.get("unique", True) and _is_queued(queue, behavior, kwargs):
            return
        queue.append((behavior, kwargs))

    def execute(self, eptm):
        """
        Executes the events present in the `self.current` deque.
        """

        while self.current:
            (behavior, kwargs) = self.current.popleft()
            if "face_id" in kwargs:
                elem_id = kwargs["face_id"]
            elif "elem_id" in kwargs:
                elem_id = kwargs["elem_id"]
            else:
                elem_id = -1
            logger.debug(f"{self.clock}, {elem_id}, {behavior.__name__}")
            behavior(eptm, self, **kwargs)

    def update(self):
        """
        Replaces `self.current` by `self.next` and clears `self.next`.
        """
        self.clock += 1
        random.shuffle(self.next)
        self.current = self.next.copy()
        self.next.clear()


def _is_queued(queue, behavior, kwargs):
    """True if the same behavior with the same kwargs is already in queue."""
    return any(
        behavior.__name__ == tup[0].__name__ and _same_kwargs(kwargs, tup[1])
        for tup in queue
    )


def _same_kwargs(kwargs, other):
    """True if both kwargs dicts have the same keys and equal values."""
    if kwargs.keys() != other.keys():
        return False
    return all(_same_value(kwargs[key], other[key]) for key in kwargs)


def _same_value(value, other):
    """Equality that also handles arrays; uncomparable values count as different."""
    if value is other:
        return True
    try:
        return bool(np.array_equal(value, other))
    except (ValueError, TypeError):
        return False


# Default dictionary for wait function
default_wait_spec = {"n_steps": 1}


def wait(eptm, manager, **kwargs):
    """Does nothing for a number of steps n_steps."""
    wait_spec = default_wait_spec
    wait_spec.update(**kwargs)
    if kwargs["n_steps"] > 1:
        kwargs.update({"n_steps": kwargs["n_steps"] - 1})
        manager.next.append((wait, wait_spec))
