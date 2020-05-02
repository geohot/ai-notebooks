import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *

def to_one_hot(x,n):
  ret = np.zeros([n])
  if x >= 0:
    ret[x] = 1.0
  return ret

def bstack(bb):
  ret = [[x] for x in bb[0]]
  for i in range(1, len(bb)):
    for j in range(len(bb[i])):
      ret[j].append(bb[i][j])
  return [np.array(x) for x in ret]

def reformat_batch(batch, a_dim, remove_policy=False):
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
  if remove_policy:
    nY = [Y[0]]
    for i in range(3, len(Y), 3):
      nY.append(Y[i])
      nY.append(Y[i+1])
    Y = nY
  else:
    Y = [Y[0]] + Y[2:]
  return X,Y

class MuModel():
  LAYER_COUNT = 4
  LAYER_DIM = 128
  BN = False

  def __init__(self, o_dim, a_dim, s_dim=8, K=5, lr=0.001, with_policy=True):
    self.o_dim = o_dim
    self.a_dim = a_dim
    self.losses = []
    self.with_policy = with_policy

    # h: representation function
    # s_0 = h(o_1...o_t)
    x = o_0 = Input(o_dim)
    for i in range(self.LAYER_COUNT):
      x = Dense(self.LAYER_DIM, activation='elu')(x)
      if i != self.LAYER_COUNT-1 and self.BN:
        x = BatchNormalization()(x)
    s_0 = Dense(s_dim, name='s_0')(x)
    self.h = Model(o_0, s_0, name="h")

    # g: dynamics function (recurrent in state?) old_state+action -> state+reward
    # r_k, s_k = g(s_k-1, a_k)
    s_km1 = Input(s_dim)
    a_k = Input(self.a_dim)
    x = Concatenate()([s_km1, a_k])
    for i in range(self.LAYER_COUNT):
      x = Dense(self.LAYER_DIM, activation='elu')(x)
      if i != self.LAYER_COUNT-1 and self.BN:
        x = BatchNormalization()(x)
    s_k = Dense(s_dim, name='s_k')(x)
    r_k = Dense(1, name='r_k')(x)
    self.g = Model([s_km1, a_k], [r_k, s_k], name="g")

    # f: prediction function -- state -> policy+value
    # p_k, v_k = f(s_k)
    x = s_k = Input(s_dim)
    for i in range(self.LAYER_COUNT):
      x = Dense(self.LAYER_DIM, activation='elu')(x)
      if i != self.LAYER_COUNT-1 and self.BN:
        x = BatchNormalization()(x)
    v_k = Dense(1, name='v_k')(x)

    if self.with_policy:
      p_k = Dense(self.a_dim, name='p_k')(x)
      self.f = Model(s_k, [p_k, v_k], name="f")
    else:
      self.f = Model(s_k, v_k, name="f")

    # combine them all
    self.create_mu(K, lr)

  def ht(self, o_0):
    return self.h.predict(np.array(o_0)[None])[0]

  def gt(self, s_km1, a_k):
    r_k, s_k = self.g.predict([s_km1[None], to_one_hot(a_k, self.a_dim)[None]])
    return r_k[0][0], s_k[0]

  def ft(self, s_k):
    if self.with_policy:
      p_k, v_k = self.f.predict(s_k[None])
      return np.exp(p_k[0]), v_k[0][0]
    else:
      return [1/a_dim]*a_dim, v_k[0][0]

  def train_on_batch(self, batch):
    X,Y = reformat_batch(batch, self.a_dim, not self.with_policy)
    l = self.mu.train_on_batch(X,Y)
    self.losses.append(l)
    return l

  def create_mu(self, K, lr):
    self.K = K
    # represent
    o_0 = Input(self.o_dim, name="o_0")
    s_km1 = self.h(o_0)

    a_all, mu_all, loss_all = [], [], []

    def softmax_ce_logits(y_true, y_pred):
      return tf.nn.softmax_cross_entropy_with_logits_v2(y_true, y_pred)

    # run f on the first state
    if self.with_policy:
      p_km1, v_km1 = self.f([s_km1])
      mu_all += [v_km1, p_km1]
      loss_all += ["mse", softmax_ce_logits]
    else:
      v_km1 = self.f([s_km1])
      mu_all += [v_km1]
      loss_all += ["mse"]

    for k in range(K):
      a_k = Input(self.a_dim, name="a_%d" % k)
      a_all.append(a_k)

      r_k, s_k = self.g([s_km1, a_k])

      # predict + store
      if self.with_policy:
        p_k, v_k = self.f([s_k])
        mu_all += [v_k, r_k, p_k]
        loss_all += ["mse", "mse", softmax_ce_logits]
      else:
        v_k = self.f([s_k])
        mu_all += [v_k, r_k]
        loss_all += ["mse", "mse"]
      
      # passback
      s_km1 = s_k

    mu = Model([o_0] + a_all, mu_all)
    mu.compile(Adam(lr), loss_all)
    self.mu = mu

