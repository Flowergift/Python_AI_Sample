import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print("x_train:%s y_train:%s x_test:%s y_test:%s "%( 
      x_train.shape, y_train.shape, x_test.shape, y_test.shape))