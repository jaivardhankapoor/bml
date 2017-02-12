import edward as ed
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(-3,3,num=50)

p_prior = ed.e
p = ed.Normal(mu=tf.Variable(tf.zeros([1])), sigma=tf.nn.sftplus)
