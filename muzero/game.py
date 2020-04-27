import numpy as np
import random

class Game():
  def __init__(self, env, discount=0.95):
    self.env = env
    self.observations = []
    self.history = []
    self.rewards = []
    self.policies = []
    self.discount = discount
    self.done = False
    self.observation = env.reset()

  def terminal(self):
    return self.done

  def apply(self, a_1, p=None):
    self.observations.append(np.copy(self.observation))
    self.observation, r_1, done, _ = self.env.step(a_1)

    self.history.append(a_1)
    self.rewards.append(r_1)
    self.policies.append(p)

    self.done = done

  def act_with_policy(self, policy):
    act = np.random.choice(list(range(len(policy))), p=policy)
    self.apply(act, policy)

  def make_image(self, i):
    return self.observations[i]

  def make_target(self, state_index, num_unroll_steps):
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      value = 0
      for i, reward in enumerate(self.rewards[current_index:]):
        value += reward * self.discount**i
        
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      targets.append((value, last_reward, self.policies[current_index]))
    return targets 

class ReplayBuffer():
  def __init__(self, window_size, batch_size, num_unroll_steps):
    self.window_size = window_size
    self.batch_size = batch_size
    self.num_unroll_steps = num_unroll_steps
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + self.num_unroll_steps],
             g.make_target(i, self.num_unroll_steps))
             for (g, i) in game_pos]

  def sample_game(self):
    return random.choice(self.buffer)

  def sample_position(self, game):
		# have to do -5 to allow enough actions
    return random.randint(0, len(game.history)-1-self.num_unroll_steps)



