import datetime
import tensorflow as tf

TEST_SIZE = 0.2
memory_window_length = 2
T = 50


log_dir = '../logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def train_model(model, x_train, y_train):

    # Fit to find a reward model
    model.fit([x_train[:, i] for i in range(x_train.shape[1])], y_train,
              epochs=5, callbacks=tensorboard_callback)

    model.save_weights('out/model_weights/toy_centered_array/test_500k_sigmoid.h5')



