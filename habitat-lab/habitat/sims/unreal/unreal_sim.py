#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any

import numpy as np

# import pyrobot
from gym import Space, spaces

from habitat.core.registry import registry
from habitat.core.simulator import (
    SemanticSensor,
    DepthSensor,
    RGBSensor,
    SensorSuite,
    Simulator,
)
from habitat.core.utils import center_crop, try_cv2_import

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.sims.unreal.unreal_link import UnrealLink

import asyncio


cv2 = try_cv2_import()


# TODO send server message to resize?
def _resize_observation(obs, observation_space, config):
    if obs.shape != observation_space.shape:
        if (
            config.center_crop is True
            and obs.shape[0] > observation_space.shape[0]
            and obs.shape[1] > observation_space.shape[1]
        ):
            obs = center_crop(obs, observation_space)

        else:
            obs = cv2.resize(
                obs, (observation_space.shape[1], observation_space.shape[0])
            )
    return obs


# What scale are we using?
MM_IN_METER = 1000  # millimeters in a meter


@registry.register_sensor
class UnrealRGBSensor(RGBSensor):
    def __init__(self, config):
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.height, self.config.width, 3),
            dtype=np.uint8,
        )

    def get_observation(self, link: UnrealLink, *args: Any, **kwargs: Any):
        # TODO link.send_packet
        print(f"Wanted RGB sensor observation? {self}")
        """obs = robot_obs.get(self.uuid, None)

        assert obs is not None, "Invalid observation for {} sensor".format(
            self.uuid
        )

        obs = _resize_observation(obs, self.observation_space, self.config)

        return obs
        """


@registry.register_sensor
class UnrealDepthSensor(DepthSensor):
    min_depth_value: float
    max_depth_value: float

    # TODO define min/max
    def __init__(self, config):
        if config.normalize_depth:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.min_depth
            self.max_depth_value = config.max_depth

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.height, self.config.width, 1),
            dtype=np.float32,
        )

    def get_observation(self, link: UnrealLink, *args: Any, **kwargs: Any):
        # TODO link.send_packet
        print(f"Wanted Depth sensor observation? {self}")
        """
        obs = robot_obs.get(self.uuid, None)

        assert obs is not None, "Invalid observation for {} sensor".format(
            self.uuid
        )

        obs = _resize_observation(obs, self.observation_space, self.config)

        obs = obs / MM_IN_METER  # convert from mm to m

        obs = np.clip(obs, self.config.min_depth, self.config.max_depth)
        if self.config.normalize_depth:
            # normalize depth observations to [0, 1]
            obs = (obs - self.config.min_depth) / (
                self.config.max_depth - self.config.min_depth
            )

        obs = np.expand_dims(obs, axis=2)  # make depth observations a 3D array

        return obs
        """


@registry.register_sensor
class UnrealSemanticSensor(SemanticSensor):
    def __init__(self, config) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.height, self.config.width, 1),
            dtype=np.int32,
        )

    def get_observation(self, link: UnrealLink, *args: Any, **kwargs: Any):
        # TODO link.send_packet
        print(f"Wanted Semantic sensor observation? {self}")

        """obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        # make semantic observation a 3D array
        if isinstance(obs, np.ndarray):
            obs = obs[..., None].astype(np.int32)
        else:
            obs = obs[..., None]
        return obs
        """


class UnknownSetting(Exception):
    pass


class IncompatibleSetting(Exception):
    pass


@registry.register_simulator(name="Unreal-Simulator-v0")
class UnrealSimulator(Simulator):
    r"""Simulator wrapper over Unreal.

    Establishes a TCP connection to the unreal server

    Args:
        config: configuration for initializing the Unreal object.
    """

    _config = {
        "character_height": 95,
        "max_slope": 20,
        "max_step_height": 40,
        "max_danger_distance": 20,
        "turn_amount": 69,
        "move_amount": 420,
        "capture_resolution": "640x480",
        "capture_sensors": "FinalImage,ObjectMask,WorldNormal,SceneDepth",
    }  # TODO define default values, units

    def __init__(self, config: "DictConfig") -> None:
        """TODO CONFIG:
        - Character Height
        - Max Slope
        - Max Step Height
        - Max Danger Distance
        - Turn Amount?
        - Move Amount?
        - Capture Resolution 640x480
        - Sensors to Capture
        """
        try:
            for k, v in config.items():
                if k not in self._config:
                    # ERROR unused settings?
                    raise UnknownSetting(k)
                else:
                    # override defaults
                    self._config[k] = v
        except UnknownSetting:
            print(f"unknown setting given to initializer ({k})")

        from unreal_link import UnrealLink

        self.client = UnrealLink()  # TODO specify IP to reach home

        async def submit_settings():
            try:
                for k, v in self._config.items():
                    response = await self.client.send_packet(f"{k} {v}")
                    if response != "OK":
                        raise IncompatibleSetting
            except IncompatibleSetting:
                print(f"Couldn't register setting {k} with value {v}")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(submit_settings)

        self.sensors = []
        for s in self._config["capture_sensors"].split(","):
            sensor_type = registry.get_sensor(s)

            assert sensor_type is not None, "invalid sensor type {}".format(s)
            self.sensors.append(sensor_type(s))
        self._sensor_suite = SensorSuite(self.sensors)

        self._action_space = spaces.Discrete(
            len(
                self.sim_config.agents[
                    self.habitat_config.default_agent_id
                ].action_space
            )
        )
        print(f"action space: {self._action_space}")
        return

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def reset(self):
        # TODO implement
        """observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )
        return observations
        """

    def step(self, action, action_params):
        r"""TODO implement"""

        print(
            f"Attempting to execute action {action} with params {action_params}"
        )
        # check if action supported
        # self.client.send_packet

        """if action in self._robot_config.base_actions:
            getattr(self._robot.base, action)(**action_params)
        elif action in self._robot_config.camera_actions:
            getattr(self._robot.camera, action)(**action_params)
        else:
            raise ValueError("Invalid action {}".format(action))

        observations = self._sensor_suite.get_observations(
            robot_obs=self.get_robot_observations()
        )

        return observations
        """

    def render(self, mode: str = "rgb") -> Any:
        # TODO
        print(f"Attempting to render with mode {mode}")

        observations = self._sensor_suite.get_observations(link=self.client)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)

        return output

    def get_agent_state(
        self, agent_id: int = 0, base_state_type: str = "odom"
    ):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        # TODO implement
        # return state
