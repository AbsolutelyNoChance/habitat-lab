#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    AgentState,
)

import quaternion

from habitat.core.dataset import Episode
from habitat.core.utils import center_crop, try_cv2_import

from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.sims.unreal.unreal_link import UnrealLink

from habitat.sims.unreal.observations import Observations

import asyncio

from enum import Enum

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


class UnrealSimActions(Enum):
    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3


@registry.register_sensor
class UnrealRGBSensor(RGBSensor):
    unreal_buffer_name = "FinalImage"

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
        # print(f"Wanted RGB sensor observation? {self}")
        return Observations[self.unreal_buffer_name]
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
    unreal_buffer_name = "Depth"

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
        # print(f"Wanted Depth sensor observation? {self}")
        return Observations[self.unreal_buffer_name]
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
    unreal_buffer_name = "ObjectMask"

    def __init__(self, config) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.config.height, self.config.width, 3),
            dtype=np.uint8,
        )

    def get_observation(self, link: UnrealLink, *args: Any, **kwargs: Any):
        # TODO link.send_packet
        # print(f"Wanted Semantic sensor observation? {self}")
        return Observations[self.unreal_buffer_name]

        """obs = cast(Optional[VisualObservation], sim_obs.get(self.uuid, None))
        check_sim_obs(obs, self)
        # make semantic observation a 3D array
        if isinstance(obs, np.ndarray):
            obs = obs[..., None].astype(np.int32)
        else:
            obs = obs[..., None]
        return obs
        """


@registry.register_simulator(name="Unreal-Simulator-v0")
class UnrealSimulator(Simulator):
    r"""Simulator wrapper over Unreal.

    Establishes a TCP connection to the unreal server

    Args:
        config: configuration for initializing the Unreal object.
    """

    current_level: int = 0

    def __init__(self, config: "DictConfig") -> None:
        """TODO CONFIG:
        - Character Height == agent.height
        - Max Slope == UnrealConfig TODO define unit
        - Max Step Height == UnrealConfig TODO define unit
        - Max Danger Distance == agent.radius
        - Turn Amount? == SimulatorConfig.turn_angle in degrees
        - Move Amount? == SimulatorConfig.forward_step_size in metres
        - Capture Resolution 640x480 == sim_sensors.height/width
        - Sensors to Capture == defined on each agent, RGBDS currently
        """

        print(f"config: {config}")

        self._config = config

        self.client = UnrealLink()  # HOME "100.75.90.104"
        self.client.connect_server()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.submit_settings(self._config))

        sim_sensors = []
        for agent_config in self._config.agents.values():
            for sensor_cfg in agent_config.sim_sensors.values():
                sensor_type = registry.get_sensor(sensor_cfg.type)

                assert (
                    sensor_type is not None
                ), "invalid sensor type {}".format(sensor_cfg.type)
                sim_sensors.append(sensor_type(sensor_cfg))
        self._sensor_suite = SensorSuite(sim_sensors)

        self.target_location = loop.run_until_complete(
            self.client.begin_simulation()
        )

        self.reset()

        # TODO idk how to use this
        """self._action_space = spaces.Discrete(
            len(
                self.sim_config.agents[
                    self.habitat_config.default_agent_id
                ].action_space
            )
        )
        print(f"action space: {self._action_space}")
        """
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
        print("Resetting environment")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.reset_environment())

        return self._sensor_suite.get_observations(link=self.client)

    def step(self, action):
        r"""TODO implement"""

        # print(f"Attempting to execute action {action}")
        # check if action supported
        # self.client.send_packet

        loop = asyncio.get_event_loop()
        if action is None:
            # according to habitat_sim given action can be null?
            loop.run_until_complete(self.client.capture_observation())
        else:
            loop.run_until_complete(self.client.execute_action(action))

        return self._sensor_suite.get_observations(link=self.client)

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

        # Trigger a recapture
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.capture_observation())

        observations = self._sensor_suite.get_observations(link=self.client)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)

        return output

    def reconfigure(
        self,
        habitat_config: Any,
        ep_info: Optional[Episode] = None,
        should_close_on_new_scene: bool = True,
    ) -> None:
        # print(f"new config: {habitat_config}")
        # print(f"{should_close_on_new_scene=}")
        # try:
        #    print(f"{ep_info=}")
        # except:
        #    print("no ep info")

        self._config = habitat_config

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.submit_settings(self._config))

        self.reset()

    def previous_step_collided(self) -> bool:
        r"""Whether or not the previous step resulted in a collision

        :return: :py:`True` if the previous step resulted in a collision,
            :py:`False` otherwise
        """

        #TODO define list of possible reasons
        return Observations["_previous_step_reset"] and Observations["_previous_step_reset_reason"] == "Hit obstacle"

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        loop = asyncio.get_event_loop()
        distance = loop.run_until_complete(
            self.client.query_geodesic_distance(position_a, position_b)
        )

        # print(
        #    f"Computed distance from {position_a} to {position_b} = {distance}"
        # )

        return distance

    def distance_to_closest_obstacle(
        self, position: np.ndarray, max_detection_radius: float
    ):
        loop = asyncio.get_event_loop()
        distance = loop.run_until_complete(
            self.client.query_closest_obstacle_distance(
                position, max_detection_radius
            )
        )

        return distance

    def get_agent_state(
        self, agent_id: int = 0, base_state_type: str = "odom"
    ):
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )

        location = Observations["_location"]
        rotation = Observations["_rotation"]

        # state = AgentState(
        #    np.asarray([0, 0, 0]), quaternion.from_euler_angles([0, 0, 0])
        # )
        # type specs state that List[float] is acceptable location format, but it later tries to do current_pos - prev_pos and that breaks.....
        state = AgentState(
            np.array(location, dtype=np.float32),
            quaternion.quaternion(*rotation),
        )
        # TODO implement, this is just a temporary fix

        return state

    def step_physics(self, delta):
        pass
        # TODO habitat always call this even if step_physics is set to false?

    def seed(self, seed: int) -> None:
        # TODO do something with the seed? for the procedural generation
        pass
