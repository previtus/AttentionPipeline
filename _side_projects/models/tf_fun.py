import numpy as np
import tensorflow as tf

y = tf.constant([[1.0,2.0,3.0,4.0,5.0,6.0][0]])

x = tf.constant([[1.0,2.0,3.0,4.0,5.0,6.0]])
x = tf.transpose(x)

print y
print x

group_by = tf.constant(3)
n = tf.size(x) / group_by

idx = tf.range(n)
idx = tf.reshape(idx, [-1, 1])  # Convert to a len(yp) x 1 matrix.
idx = tf.tile(idx, [1, group_by])  # Create multiple columns.
idx = tf.reshape(idx, [-1])  # Convert back to a vector.

a = tf.size(x)
b = tf.size(idx)
test = tf.segment_mean(x,idx)

sess = tf.Session()
print(sess.run([idx]))
print(sess.run([n]))
print(sess.run([x]))
print(sess.run([a,b]))
print(sess.run([test]))
