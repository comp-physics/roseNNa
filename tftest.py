import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
# class MyModel(Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.d1 = Dense(128, activation='relu')
#         self.d2 = Dense(10)

#     def call(self, x):
#         x = self.conv1(x)
#         x = self.flatten(x)
#         x = self.d1(x)
#         return self.d2(x)

# # Create an instance of the model
# model = MyModel()
# print(model.layers[1].shape)

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(10)
  ])

#   model.compile(optimizer='adam',
#                 loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()
model.load_weights("model.ckpt")

# Display the model's architecture
print(model.layers[0].weights[0].numpy().shape)
print(model.layers[1].weights[0].numpy().shape)


