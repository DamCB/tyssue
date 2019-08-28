"""
Event management module
=======================


"""

import logging
import random
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            behavior(sheet, manager, **kwargs)

        this function itself might populate the managers next deque

        Parameters
        ----------
        behavior : function
        kwargs : dict defaults to {}
            keywords arguments to the behavior function
            if `"face_id"` is in the kwargs dictionnary,
            the face with this id is used.
        """
        if "face_id" in kwargs:
            elem_id = kwargs["face_id"]
        elif "elem_id" in kwargs:
            elem_id = kwargs["elem_id"]
        else:
            elem_id = -1

        for tup in self.next:
            unique = kwargs.get("unique", True)
            if "face_id" in tup[1]:
                if (elem_id == tup[1]["face_id"]) and (
                    behavior.__name__ == tup[0].__name__ and (unique)
                ):
                    return
            elif "elem_id" in tup[1]:
                if (elem_id == tup[1]["elem_id"]) and (
                    behavior.__name__ == tup[0].__name__(unique)
                ):
                    return
        self.next.append((behavior, kwargs))

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
default_wait_spec = {"n_steps": 1}


def wait(eptm, manager, **kwargs):
    """Does nothing for a number of steps n_steps
    """
    wait_spec = default_wait_spec
    wait_spec.update(**kwargs)
    if kwargs["n_steps"] > 1:
        kwargs.update({"n_steps": kwargs["n_steps"] - 1})
        manager.next.append((wait, wait_spec))
