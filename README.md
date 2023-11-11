# flower_opacus
Federated learning with differential privacy using Flower and Opacus packages in Pytorch


File description:

The client[x].py files each represent a uniuqe device in the simulation.
The server.py file is the centralized server to which each clients send their model parameters and metrics
the pytorch.ipynb file contains the same setup as the federated learning, but in a centralized 'vanilla' approach. this allows for easy comparison

the files with 'dp" in their name are the ones that apply differential privacy to the model parameters. Also here it includes one in federated setup and another for centralized approach

-----------------------
HOW TO RUN

start the server by using command python server.py
run each client[x].py file from their own terminal by using command python client[x].py 8080

wait for the model to complete training and see the evaluation metrics printed in the server terminal
