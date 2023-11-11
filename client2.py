import flwr as fl

from clients import FlowerClient
from clients_pytorch import FlowerClient
from clients_dp_pytorch import FlowerClient
# Start Flower client

fl.client.start_numpy_client(
        server_address="localhost:"+str(8080),
        client=FlowerClient(1),
        grpc_max_message_length = 1024*1024*1024
)
