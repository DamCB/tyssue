"""
Event management module
=======================


"""

import logging
import pandas as pd
import warnings
import random
from collections import deque, namedtuple
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EventManager:
    """
    Behavior management class based on two deques, the current and next one.



    """

    def __init__(self, element, logfile=None):
        """Creates an events class

        Parameters
        ----------
        element : str
           element on which the events occur, e.g face or cell
        logfile : str, default None
           if logfile is not None, will create a logging handler
           for this file where each event will be logged



        """
        self.current = deque()
        self.next = deque()
        self.element = element
        self.current.append((wait, {'face_id': -1, 'n_steps': 1}))
        self.clock = 0
        if logfile is not None:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.info("# Started logging at %s", datetime.now().isoformat())
            logger.info(f"time, {self.element} index, event")

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
            behavior(sheet, manager, elem_id,
                     *args, **kwargs)

        this function itself might populate the managers next deque

        Parameters
        ----------
        behavior : function
        elem_id : int, default -1
            the id of the affected element, leave to
            to -1 if the behavior is not element specific
        args : tuple, defaults to ()
            extra arguments to the behavior function
        kwargs : dict defaults to {}
            extra keywords arguments to the behavior function

        """
        elem_id = kwargs['face_id']
        for tup in self.next:
            if (elem_id == tup[1]) and (behavior.__name__ == tup[0].__name__):
                return
        self.next.append((behavior, kwargs))

    def execute(self, eptm):
        """
        Executes the events present in the `self.current` deque.
        """

        while self.current:
            (behavior, kwargs) = self.current.popleft()
            elem_id = kwargs['face_id']
            logger.info(f"{self.clock}, {elem_id}, {behavior.__name__}")
            behavior(eptm, self, **kwargs)

    def update(self):
        """
        Replaces `self.current` by `self.next` and clears `self.next`.
        """
        random.shuffle(self.next)
        self.current = self.next.copy()
        self.next.clear()


# Default dictionary for wait function
default_wait_spec = {'n_steps': 1}


def wait(eptm, manager, **kwargs):
    """Does nothing for a number of steps n_steps
    """
    wait_spec = default_wait_spec
    wait_spec.update(**kwargs)
    elem_id = kwargs['face_id']
    if kwargs['n_steps'] > 1:
        kwargs.update({
                      'n_steps': kwargs['n_steps']-1})
        manager.next.append(("wait", wait_spec))
