import tensorflow as tf
import flwr as fl

import pandas as pd
import numpy as np
from data import load_data
from model import Model


import argparse

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()
model = Model()


class StoreClient(fl.client.NumPyClient):
    def __init__(
        self,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
    ) -> None:
        super().__init__()
        self.model = model.model
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            min_delta=100,
            restore_best_weights=True,
        )
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=100,
            epochs=100,
            callbacks=[early_stopping],
            validation_data=(x_valid, y_valid))

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "mean_squared_error": history.history["mean_squared_error"][0],
            "val_loss": history.history["val_loss"][0],
            "val_mean_squared_error": history.history["val_mean_squared_error"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mean_squared_error, mean_absolute_error = self.model.evaluate(
            self.x_test, self.y_test
        )
        return (
            loss,
            len(self.x_test),
            {
                "mean_squared_error": float(mean_squared_error),
                "mean_absolute_error": mean_absolute_error,
            },
        )





def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client", type=int, choices=range(1, 46), required=True)
    args = parser.parse_args()
    print(str(args.client))

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(str(args.client))

    model = Model()

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=StoreClient(model, X_train, y_train, X_valid, y_valid, X_test, y_test),
    )


if __name__ == "__main__":
    main()
