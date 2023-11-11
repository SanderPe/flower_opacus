import flwr as fl


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):


    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)


        # Weighing metrics of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        recalls = [r.metrics["recall"] * r.num_examples for _, r in results]
        precisions = [r.metrics["precision"] * r.num_examples for _, r in results]
        f1 = [r.metrics["f1"] * r.num_examples for _, r in results]

        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metrics
        aggregated_accuracy = sum(accuracies) / sum(examples)
        aggregated_recall = sum(recalls) / sum(examples)
        aggregated_precision = sum(precisions) / sum(examples)
        aggregated_f1 = sum(f1) / sum(examples)


        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        print(f"Round {server_round} recall aggregated from client results: {aggregated_recall}")
        print(f"Round {server_round} precision aggregated from client results: {aggregated_precision}")
        print(f"Round {server_round} f1 score aggregated from client results: {aggregated_f1}")
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy, "recall" : aggregated_recall, "precision" : aggregated_precision, "f1" : aggregated_f1}


    def aggregate_fit(
            self,
            server_round,
            results,
            failures,
    ):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:



            return aggregated_parameters, aggregated_metrics

# Create strategy and run server
strategy = AggregateCustomMetricStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(8080) ,
        config=fl.server.ServerConfig(num_rounds=20) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy,

)

