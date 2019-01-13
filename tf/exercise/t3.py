import tensorflow as tf

a = tf.constant([1, 2], name='a', dtype=tf.float32)
b = tf.constant([2.0, 3.0], name='b')
result = tf.add(a, b, name='add')

print(result)
result2 = tf.constant([1, 2], name='a', dtype=tf.float32) + tf.constant([2.0, 3.0], name='b')
print(result2)
print(a,b)
print(tf.Session().run(result))
sess = tf.Session()
with sess.as_default():
    print(result.eval())


sess1 = tf.Session()
print(result.eval(session=sess1))

tf.InteractiveSession()
print result.eval()

config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
tf.InteractiveSession(config=config)