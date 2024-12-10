import tensorflow as tf
import time
from cifar10_model import create_shallow_model
from utils import load_and_preprocess_data
from plot_results import plot_results  

x_train, y_train, x_test, y_test = load_and_preprocess_data()

optimizers = {
    "SGD with Warm Restarts": tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.01, first_decay_steps=1000)
    ),
    "NAG": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "Nadam": tf.keras.optimizers.Nadam(learning_rate=0.001),
    "Exponential Decay": tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.9
        )
    ),
    "Step Decay": tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[1000, 2000],  
            values=[0.01, 0.005, 0.001]  
        )
    )
}

histories = []
labels = []

for name, optimizer in optimizers.items():
    model = create_shallow_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=64, verbose=2)
    training_time = time.time() - start_time
    print(f"{name} Training Time: {training_time:.2f} seconds")

    histories.append(history)
    labels.append(name)


plot_results(histories, labels)
