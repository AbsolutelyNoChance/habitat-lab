from habitat.core.utils import Singleton

from habitat.core.simulator import VisualObservation

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import attr

import base64

import json

import numpy as np


@attr.s(auto_attribs=True, slots=True)
class ObservationsSingleton(metaclass=Singleton):
    buffers_to_get: List[str] = attr.ib(init=False, factory=list)
    buffers: Dict[str, VisualObservation] = attr.ib(init=False, factory=dict)

    def set_buffers(self, buffers: List[str]):
        # TODO implement this
        self.buffers_to_get = buffers

    def parse_buffers(self, obj):
        self.buffers = {}
        for key, value in obj.items():
            print(f"Got buffer {key}")
            image = base64.b64decode(value)

            f = open(f"{key}.png", "wb")
            f.write(image)
            f.close()

            self.buffers[key] = np.asarray(image)

        print(f"Got observations: {', '.join([k for k in obj.keys()])}")

    def __getattr__(self, name):
        return self.buffers[name]

    def __getitem__(self, name):
        return self.buffers[name]


Observations: ObservationsSingleton = ObservationsSingleton()
