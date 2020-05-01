import math
import random
import numpy as np

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

class Node(object):
  def __init__(self, prior: float):
    self.visit_count = 0
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0
    self.to_play = -1

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

pb_c_base = 19652
pb_c_init = 1.25

discount = 0.95
root_dirichlet_alpha = 0.25
root_exploration_fraction = 0.25

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self):
    self.maximum = -float('inf')
    self.minimum = float('inf')

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node, min_max_stats=None) -> float:
  pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    if min_max_stats is not None:
      value_score = child.reward + discount * min_max_stats.normalize(child.value())
    else:
      value_score = child.reward + discount * child.value()
  else:
    value_score = 0

  #print(prior_score, value_score)
  return prior_score + value_score

def select_child(node: Node, min_max_stats=None):
  out = [(ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items()]
  smax = max([x[0] for x in out])
  # this max is why it favors 1's over 0's
  _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
  return action, child

def mcts_search(m, observation, num_simulations=10, minimax=True):
  # init the root node
  root = Node(0)
  root.hidden_state = m.ht(observation)
  if minimax:
    root.to_play = observation[-1]
  policy, value = m.ft(root.hidden_state)

  # expand the children of the root node
  for i in range(policy.shape[0]):
    root.children[i] = Node(policy[i])
    root.children[i].to_play = -root.to_play

  # add exploration noise at the root
  actions = list(root.children.keys())
  noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
  frac = root_exploration_fraction
  for a, n in zip(actions, noise):
    root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac

  # run_mcts
  #min_max_stats = MinMaxStats()
  for _ in range(num_simulations):
    history = []
    node = root
    search_path = [node]

    # traverse down the tree according to the ucb_score 
    while node.expanded():
      #action, node = select_child(node, min_max_stats)
      action, node = select_child(node)
      history.append(action)
      search_path.append(node)

    # now we are at a leaf which is not "expanded", run the dynamics model
    parent = search_path[-2]
    node.reward, node.hidden_state = m.gt(parent.hidden_state, history[-1])

    # use the model to estimate the policy and value, use policy as prior
    policy, value = m.ft(node.hidden_state)
    #print(history, value)

    # create all the children of the newly expanded node
    for i in range(policy.shape[0]):
      node.children[i] = Node(prior=policy[i])
      node.children[i].to_play = -node.to_play

    # update the state with "backpropagate"
    for bnode in reversed(search_path):
      if minimax:
        bnode.value_sum += value if root.to_play == bnode.to_play else -value
      else:
        bnode.value_sum += value
      bnode.visit_count += 1
      #min_max_stats.update(node.value())
      value = bnode.reward + discount * value

  # output the final policy
  visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
  visit_counts = [x[1] for x in sorted(visit_counts)]
  av = np.array(visit_counts).astype(np.float64)
  policy = softmax(av)
  return policy, root

def print_tree(x, hist=[]):
  if x.visit_count != 0:
    print("%3d %4d %-16s %8.4f %4d" % (x.to_play, x.visit_count, str(hist), x.value(), x.reward))
  for i,c in x.children.items():
    print_tree(c, hist+[i])

def get_action_space(K, n):
  def to_one_hot(x,n):
    ret = np.zeros([n])
    ret[x] = 1.0
    return ret
  import itertools
  aopts = list(itertools.product(list(range(n)), repeat=K))
  aoptss = np.array([[to_one_hot(x, n) for x in aa] for aa in aopts])
  aoptss = aoptss.swapaxes(0,1)
  aoptss = [aoptss[x] for x in range(K)]
  return aopts,aoptss

# TODO: this is naive search, replace with MCTS
aspace = {}
def naive_search(m, o_0, debug=False, T=1):
  K,n = m.K, m.a_dim
  if (K,n) not in aspace:
    aspace[(K,n)] = get_action_space(K, n)
  aopts,aoptss = aspace[(K,n)]

  # concatenate the current state with every possible action
  o_0s = np.repeat(np.array(o_0)[None], len(aopts), axis=0)
  ret = m.mu.predict([o_0s]+aoptss)
  v_s = ret[-3]
  
  minimum = min(v_s)
  maximum = max(v_s)
  v_s = (v_s - minimum) / (maximum - minimum)
  
  # group the value with the action rollout that caused it
  v = [(v_s[i][0], aopts[i]) for i in range(len(v_s))]
  if debug:
    print(sorted(v, reverse=True))
  
  av = [0] * n
  for vk, ak in v:
    av[ak[0]] += vk

  av = np.array(av).astype(np.float64) / T
  policy = softmax(av)
  return policy

def get_values(m, o_0):
	hidden_state = m.ht(o_0)
	vs = []
	for n in range(m.a_dim):
		_, ht2 = m.gt(hidden_state, n)
		_, v = m.ft(ht2)
		vs.append(v)
	return vs


