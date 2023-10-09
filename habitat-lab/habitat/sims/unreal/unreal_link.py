# TCP client implementation
import socket
import struct
import json
import base64

from habitat.sims.unreal.observations import Observations

from habitat.sims.unreal.actions import UnrealSimActions

from omegaconf import OmegaConf


class UnrealLink:
    def __init__(self, ip="127.0.0.1") -> None:
        self.ip = ip  # "100.75.90.104"  # tailscale home machine
        self.port = 8890
        self.packet_size = 4096

    def connect_server(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((self.ip, self.port))
        self.client.settimeout(5)

    def close_connection(self):
        self.client.close()

    async def __receive_packet(self):
        try:
            size_packet = self.client.recv(4)
            size = struct.unpack("<i", size_packet)[0]

            next_recv = min(self.packet_size, size)
            response = self.client.recv(next_recv)
            while len(response) < size:
                next_rcv = min(self.packet_size, size - len(response))
                response += self.client.recv(next_rcv)

            decoded = response.decode()
            print(f"Received {len(response)} bytes")

            return decoded
        except Exception as e:
            print(e)

    async def __send_packet(self, payload):
        self.client.send(payload.encode())

        # always await a response
        response = await self.__receive_packet()

        return response

    async def execute_action(self, action):
        # TODO error check? make new json field to detect errors or stop?
        action_name = UnrealSimActions.get_unreal_action(action)

        print(f"Executing action {action_name}")

        observation = await self.__send_packet(f"action {action_name}")

        # Testing if it's a json... yeah
        if observation[0] == "{":
            obj = json.loads(observation)
            Observations.parse_buffers(obj)
        else:
            print(observation)

    async def capture_observation(self):
        # TODO error check? make new json field to detect errors or stop?
        observation = await self.__send_packet("capture")

        # Testing if it's a json... yeah
        if observation[0] == "{":
            obj = json.loads(observation)
            Observations.parse_buffers(obj)
        else:
            print(observation)

    async def submit_settings(self, config):
        result = await self.__send_packet(
            json.dumps(OmegaConf.to_container(config))
        )

        if result == "OK":
            pass
        else:
            print(f"Unreal server didn't accept the settings! {result}")
            exit()

    async def begin_simulation(self):
        # TODO error check? make new json field to detect errors or stop?
        print(f"Beginning the simulation")

        result = await self.__send_packet("begin_sim")

        try:
            target_location = [float(i) for i in result.split(" ")]
            return target_location
        except Exception as e:
            print(f"Unreal server didn't start the simulation! {result}")
            exit()

    async def query_geodesic_distance(self, point_a, point_b):
        # TODO error check? make new json field to detect errors or stop?
        queried_distance = await self.__send_packet(
            f"geodesic_distance {point_a} {point_b}"
        )

        try:
            distance = float(queried_distance)
            if distance == -1.0:
                raise Exception("Invalid path!")
            return distance
        except Exception as e:
            print(f"Could not compute geodesic distance! {e}")
