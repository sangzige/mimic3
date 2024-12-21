import warnings

import numpy as np

from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch

warnings.filterwarnings(action='ignore', message='Setting attributes')


def main():
    # https://keras.io/layers/recurrent/
    num_samples, time_steps, input_dim, output_steps = 50_000, 10, 1, 1

    # Definition of the model.
    # NOTE: If you choose the Keras backend with input_dim>1, you have
    # to set the value here too (in the constructor).


    model_pytorch = NBeatsPytorch(backcast_length=time_steps, forecast_length=output_steps,
                                  stack_types=(NBeatsPytorch.GENERIC_BLOCK, NBeatsPytorch.GENERIC_BLOCK),
                                  nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
                                  hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    model_pytorch.compile(loss='mae', optimizer='adam')

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    # where f = np.mean.
    x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    y = np.mean(x, axis=1, keepdims=True)

    # Split data into training and testing datasets.
    c = num_samples // 10
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
    test_size = len(x_test)

    # Train the model.

    print('Pytorch training...')
    model_pytorch.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128)

    # Save the model for later.

    model_pytorch.save('n_beats_pytorch.th')

    # Predict on the testing set (forecast).

    predictions_pytorch_forecast = model_pytorch.predict(x_test)

    np.testing.assert_equal(predictions_pytorch_forecast.shape,
                            (test_size, model_pytorch.forecast_length, output_steps))

    # Predict on the testing set (backcast).
    predictions_pytorch_backcast = model_pytorch.predict(x_test, return_backcast=True)
    np.testing.assert_equal(predictions_pytorch_backcast.shape,
                            (test_size, model_pytorch.backcast_length, output_steps))

    # Load the model.
    model_pytorch_2 = NBeatsPytorch.load('n_beats_pytorch.th')

    np.testing.assert_almost_equal(predictions_pytorch_forecast, model_pytorch_2.predict(x_test))


if __name__ == '__main__':
    main()
