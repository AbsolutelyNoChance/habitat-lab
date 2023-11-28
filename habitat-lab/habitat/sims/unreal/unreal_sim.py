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

import csv

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
from habitat.sims.unreal.actions import UnrealSimActions

from habitat.sims.unreal.observations import Observations

import asyncio

from enum import Enum

cv2 = try_cv2_import()


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

        # print(f"config: {config}")

        self._config = config

        self.client = UnrealLink()  # HOME "100.75.90.104" "100.120.227.10"
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

        # self.target_location = loop.run_until_complete(
        #    self.client.begin_simulation()
        # )

        # to keep track of DistanceClosestObstacle
        self.dco_list: List[float] = []

        self.step_count = 0
        self.dangerous_step_count = 0

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

        # TODO only count data that has more than x moves?
        # TODO do we count turns? I guess it's more time spent in potentially dangerous areas?
        # TODO only count if episodes are successful?
        # TODO eval dataset different than training dataset
        if len(self.dco_list) > 0:
            with open("dco.csv", "a") as file:
                # print(f"saved {''.join(str(self.dco_list))}")

                writer = csv.writer(file)
                writer.writerow(self.dco_list)

                self.dco_list = []

        if self.step_count > 0:
            print(
                f"Took {self.step_count} steps of which {self.dangerous_step_count} were dangerous ({float(self.dangerous_step_count)/float(self.step_count)} %)"
            )

        # print("Resetting environment")
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

            dco = self.distance_to_closest_obstacle(
                self.get_agent_state().position, 200
            )  # meters
            # print(f"computed dco: {dco}")
            self.dco_list.append(dco)

        # XXX Should we count turning steps?
        if UnrealSimActions.is_moving_action(action):
            if dco < 0.5:  # 0.5 meters, 50cm
                # We are in a dangerous location
                self.dangerous_step_count += 1
                print(
                    f"Previous action was a moving + dangerous action ({dco} meters away from nearest obstacle)"
                )
            self.step_count += 1

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

        if ep_info is not None:
            self.target_location = loop.run_until_complete(
                self.client.begin_simulation(
                    ep_info.start_position, ep_info.start_rotation
                )
            )

    def previous_step_collided(self) -> bool:
        r"""Whether or not the previous step resulted in a collision

        :return: :py:`True` if the previous step resulted in a collision,
            :py:`False` otherwise
        """

        # TODO define list of possible reasons
        return (
            Observations["_previous_step_reset"]
            and Observations["_previous_step_reset_reason"] == "Hit obstacle"
        )

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode: Optional[Episode] = None,
    ) -> float:
        # flatten position_b since it's a list of points?
        position_b = sum(position_b, [])

        loop = asyncio.get_event_loop()
        distance = loop.run_until_complete(
            self.client.query_geodesic_distance(position_a, position_b)
        )

        if distance < 0:
            # try again?
            loop = asyncio.get_event_loop()
            distance = loop.run_until_complete(
                self.client.query_geodesic_distance(position_a, position_b)
            )

        if distance < 0:
            # if still bad path??
            print(
                f"Failed to compute the geodesic distance twice from {position_a} to {position_b}"
            )
            exit()

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
