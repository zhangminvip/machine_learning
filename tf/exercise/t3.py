import tensorflow as tf

a = tf.constant([1, 2], name='a', dtype=tf.float32)
b = tf.constant([2.0, 3.0], name='b')
result = tf.add(a, b, name='add')
print(result)