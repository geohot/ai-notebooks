import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def to_one_hot(x,n):
  ret = np.zeros([n])
  ret[x] = 1.0
  return ret

def bstack(bb):
  ret = [[x] for x in bb[0]]
  for i in range(1, len(bb)):
    for j in range(len(bb[i])):
      ret[j].append(bb[i][j])
  return [np.array(x) for x in ret]

def reformat_batch(batch, a_dim):
  X,Y = [], []
  for o,a,outs in batch:
    x = [o] + [to_one_hot(x, a_dim) for x in a]
    y = []
    for ll in [list(x) for x in outs]:
      y += ll
    X.append(x)
    Y.append(y)
  X = bstack(X)
  Y = bstack(Y)
  Y = [Y[0]] + Y[2:]
  return X,Y

class MuModel():
  # the dimension of the internal state
  S_DIM = 8

  def __init__(self, o_dim, a_dim):
    self.o_dim = o_dim
    self.a_dim = a_dim
    self.losses = []

    # h: representation function
    # s_0 = h(o_1...o_t)
    x = o_0 = Input(o_dim)
    x = Dense(64)(x)
    x = Activation('elu')(x)
    s_0 = Dense(self.S_DIM, name='s_0')(x)
    self.h = Model(o_0, s_0, name="h")

    # g: dynamics function (recurrent in state?) old_state+action -> state+reward
    # r_k, s_k = g(s_k-1, a_k)
    s_km1 = Input(self.S_DIM)
    a_k = Input(a_dim)
    x = Concatenate()([s_km1, a_k])
    x = Dense(64)(x)
    x = Activation('elu')(x)
    x = Dense(64)(x)
    x = Activation('elu')(x)
    s_k = Dense(self.S_DIM, name='s_k')(x)
    r_k = Dense(1, name='r_k')(x)
    self.g = Model([s_km1, a_k], [r_k, s_k], name="g")

    # f: prediction function -- state -> policy+value
    # p_k, v_k = f(s_k)
    x = s_k = Input(self.S_DIM)
    x = Dense(32)(x)
    x = Activation('elu')(x)
    p_k = Dense(a_dim, name='p_k')(x)
    v_k = Dense(1, name='v_k')(x)
    self.f = Model(s_k, [p_k, v_k], name="f")

    # combine them all
    self.create_mu()

  def ht(self, o_0):
    return self.h.predict(np.array(o_0)[None])[0]

  def gt(self, s_km1, a_k):
    r_k, s_k = self.g.predict([s_km1[None], a_k[None]])
    return r_k[0], s_k[0]

  def ft(self, s_k):
    p_k, v_k = self.f.predict(s_k[None])
    return p_k[0], v_k[0]

  def train_on_batch(self, batch):
    X,Y = reformat_batch(batch, self.a_dim)
    l = self.mu.train_on_batch(X,Y)
    self.losses.append(l)
    return l

  def create_mu(self, K=5):
    # represent
    o_0 = Input(self.o_dim, name="o_0")
    s_km1 = self.h(o_0)

    a_all, mu_all, loss_all = [], [], []

    # run f on the first state
    p_km1, v_km1 = self.f([s_km1])
    mu_all += [v_km1, p_km1]
    loss_all += ["mse", "categorical_crossentropy"]

    for k in range(K):
      a_k = Input(self.a_dim, name="a_%d" % k)
      r_k, s_k = self.g([s_km1, a_k])

      # predict
      p_k, v_k = self.f([s_k])

      # store
      a_all.append(a_k)
      mu_all += [v_k, r_k, p_k]
      loss_all += ["mse", "mse", "categorical_crossentropy"]
      s_km1 = s_k

    mu = Model([o_0] + a_all, mu_all)
    mu.compile('adam', loss_all)
    self.mu = mu

