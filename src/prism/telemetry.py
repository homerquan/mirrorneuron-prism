"""Bounded metadata-only traces, scoped to the authenticated credential."""

import time
from collections import OrderedDict
from copy import deepcopy


class TraceStore:
    def __init__(self, capacity=256, ttl=3600):
        self.capacity = capacity
        self.ttl = ttl
        self.items = OrderedDict()

    def put(self, owner, trace):
        self._expire()
        if self.capacity:
            self.items[(owner, trace["request_id"])] = (
                time.monotonic(),
                deepcopy(trace),
            )
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)

    def get(self, owner, request_id):
        self._expire()
        item = self.items.get((owner, request_id))
        return deepcopy(item[1]) if item else None

    def _expire(self):
        cutoff = time.monotonic() - self.ttl
        while self.items and next(iter(self.items.values()))[0] < cutoff:
            self.items.popitem(last=False)
