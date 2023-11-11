# flower_opacus
Federated learning with differential privacy using Flower and Opacus packages in Pytorch


File description:

The client[x].py files each represent a uniuqe device in the simulation.
The server.py file is the centralized server to which each clients send their model parameters and metrics.


The pytorch.py file contains the same model specifications as the federated learning, but in a centralized 'vanilla' approach. this allows for easy comparison.
The clients_pytorch.py file contains the classes used to create the model in each client, clients.py does the same thing but in keras


The files with 'dp" in their name are the ones that apply differential privacy to the model parameters. Also here it includes one in federated setup and another for centralized approach.

-----------------------
HOW TO RUN

start the server by using command python server.py
run each client[x].py file from their own terminal by using command python client[x].py 8080

wait for the model to complete training and see the evaluation metrics printed in the server terminal
-----------------------
Requirements:

- Python 3.7
- Torch 1.13.1
- Opacus 1.1.2
- Flwr 1.5.0
