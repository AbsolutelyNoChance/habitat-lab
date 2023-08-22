#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_unreal():
    from habitat.sims.unreal.unreal_sim import UnrealSimulator
