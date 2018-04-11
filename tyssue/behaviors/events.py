"""
Event management module
=======================


"""

import logging
import pandas as pd
import warnings
import random
from collections import deque, namedtuple

logger = logging.getLogger(__name__)


class EventManager():
    """
    Behavior management class based on two deques, the current and next one.



    """
    def __init__(self, element):
        """Creates an events class

        Parameters
        ----------
        element : str
           element on which the events occur, e.g face or cell
        """
        self.current = deque()
        self.next = deque()
        self.element = element
        self.current.append((wait, -1, (1,), {}))


    def extend(self, events):
        """
        Add a list of events to the next deque

        Parameters
        ----------
        events : list of tuples (behavior, [elem_id, args, kwargs])
            the three last elements don't need to be passed

        """
        for event in events:
            self.append(*event)

    def append(self, behavior, elem_id=-1,
               args=None, kwargs=None):
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
        #if (behavior, elem_id) not in self.next:
        #    self.next.append((behavior, elem_id, args, kwargs))
        for tup in self.next:
            if elem_id in tup:
                return
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        self.next.append((behavior, elem_id, args, kwargs))

    def execute(self, eptm):
        """
        Executes the events present in the `self.current` deque,
        then replaces  `self.current` by `self.next`
        and clears `self.next`.
        """

        while self.current:
            (behavior, elem_id, args, kwargs) = self.current.popleft()
            logger.info(f'{self.element}: {elem_id}, behavior: {behavior.__name__}')
            behavior(eptm, self, elem_id, *args, **kwargs)

    def update(self):
        """
        Replaces 'self.current' by 'self.next'
        and clears 'self.next'.
        """
        random.shuffle(self.next)
        self.current = self.next.copy()
        self.next.clear()



def wait(eptm, manager, elem_id, n_steps):
    """Does nothing for a number of steps n_steps
    """
    if n_steps > 1:
        manager.next.append('wait', elem_id,
                            (n_steps-1,), {})
    logger.info(f'Doing nothing for {n_steps} steps')
