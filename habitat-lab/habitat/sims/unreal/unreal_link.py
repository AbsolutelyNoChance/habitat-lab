# TCP client implementation
import socket
import struct
import json
import base64

from habitat.sims.unreal.observations import ObservationsSingleton


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

    async def execute_action(self, action_name):
        # TODO error check? make new json field to detect errors or stop?
        observation = self.__send_packet(f"action {action_name}")

        # Testing if it's a json... yeah
        if observation[0] == "{":
            obj = json.loads(observation)
            ObservationsSingleton.parse_buffers(obj)

    async def capture_observation(self):
        # TODO error check? make new json field to detect errors or stop?
        observation = self.__send_packet("capture")

        # Testing if it's a json... yeah
        if observation[0] == "{":
            obj = json.loads(observation)
            ObservationsSingleton.parse_buffers(obj)

    async def submit_settings(self, config):
        try:
            for k, v in config.items():
                response = await self.client.__send_packet(f"{k} {v}")
                if response != "OK":
                    raise Exception
        except:
            print(f"Couldn't register setting {k} with value {v}")
