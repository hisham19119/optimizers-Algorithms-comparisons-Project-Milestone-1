from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Flatten, Dense # type: ignore

def create_shallow_model():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        # Dense(128, activation='relu'),
        # Dense(64, activation='relu'),
        # Hidden layers should be added like previous but according to doctor's LEC's we don't consider them to our model...
        Dense(10, activation='softmax')
    ])
    return model
