import math
import random
import numpy as np

class Node(object):
  def __init__(self, prior: float, hidden_state, reward):
    self.visit_count = 0
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = hidden_state
    self.reward = reward

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
  # this max is why it favors 1's over 0's
  _, action, child = max(out)
  return action, child

def mcts_search(m, observation, num_simulations=10):
  # init the root node
  root = Node(0, m.ht(observation), 0)
  policy, value = m.ft(root.hidden_state)

  # expand the children of the root node
  for i in range(policy.shape[0]):
    reward, hidden_state = m.gt(root.hidden_state, 0)
    root.children[i] = Node(policy[i], hidden_state, reward)

  # add exploration noise at the root
  actions = list(root.children.keys())
  noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
  frac = root_exploration_fraction
  for a, n in zip(actions, noise):
    root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac

  # run_mcts
  min_max_stats = MinMaxStats()
  for _ in range(num_simulations):
    history = []
    node = root
    search_path = [node]

    while node.expanded():
      action, node = select_child(node, min_max_stats)
      history.append(action)
      search_path.append(node)

    # now we are at a leaf which is not "expanded"
    parent = search_path[-2]
    reward, hidden_state = m.gt(parent.hidden_state, history[-1])

    policy, value = m.ft(hidden_state)
    #print(policy)
    for i in range(policy.shape[0]):
      reward, hidden_state = m.gt(node.hidden_state, 0)
      node.children[i] = Node(policy[i], hidden_state, reward)

    # update the state with "backpropagate"
    for node in reversed(search_path):
      node.value_sum += value
      node.visit_count += 1
      min_max_stats.update(node.value())
      value = node.reward + discount * value

  # output the final policy
  visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
  visit_counts = [x[1] for x in sorted(visit_counts)]
  av = np.array(visit_counts).astype(np.float64)
  policy = np.exp(av)/sum(np.exp(av))

  return policy, root

def print_tree(x, hist=[]):
  print(x.value(), x, hist)
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
def naive_search(m, o_0, debug=False):
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

  av = np.array(av).astype(np.float64)
  
  #print(av)
  
  policy = np.exp(av)/sum(np.exp(av))
  return policy

