#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import habitat

# from habitat.sims.unreal.unreal_sim import UnrealSimActions
from habitat.sims.unreal.actions import UnrealSimActions


class ForwardOnlyAgent(habitat.Agent):
    def reset(self):
        pass

    def act(self, observations):
        action = UnrealSimActions.move_forward
        return {"action": action}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        type=str,
        default="benchmark/nav/unreal.yaml",
    )
    args = parser.parse_args()

    agent = ForwardOnlyAgent()
    benchmark = habitat.Benchmark(args.task_config)
    metrics = benchmark.evaluate(agent, num_episodes=10)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
