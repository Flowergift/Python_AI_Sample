import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

model = tf.keras.Sequential([
	tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
	tf.keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)