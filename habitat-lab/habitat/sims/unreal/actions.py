#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict

import attr

from habitat.core.utils import Singleton


class _DefaultUnrealSimActions(Enum):
    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3
    look_up = 4
    look_down = 5


@attr.s(auto_attribs=True, slots=True)
class UnrealSimActionsSingleton(metaclass=Singleton):
    r"""Implements an extendable Enum for the mapping of action names
    to their integer values.

    This means that new action names can be added, but old action names cannot
    be removed nor can their mapping be altered. This also ensures that all
    actions are always contigously mapped in :py:`[0, len(HabitatSimActions) - 1]`

    This accesible as the global singleton :ref:`HabitatSimActions`
    """

    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def get_unreal_action(self, action) -> str:
        if isinstance(action, str):
            return action
        elif isinstance(action, int):
            a = [
                i
                for i in self._known_actions
                if self._known_actions[i] == action
            ]
            return a[0]  # assume only one action has that int code
        else:
            return "stop"

    def __attrs_post_init__(self):
        for action in _DefaultUnrealSimActions:
            self._known_actions[action.name] = action.value

    def has_action(self, name: str) -> bool:
        r"""Checks to see if action :p:`name` is already register

        :param name: The name to check
        :return: Whether or not :p:`name` already exists
        """

        return name in self._known_actions

    def is_moving_action(self, action) -> bool:
        r"""Checks if the given action leads to character movement.
        Moving actions: move_forward
        Non-moving: turn_left, turn_right
        """
        action_string = self.get_unreal_action(action)
        if action_string == "move_forward":
            return True
        else:
            return False

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


UnrealSimActions: UnrealSimActionsSingleton = UnrealSimActionsSingleton()
