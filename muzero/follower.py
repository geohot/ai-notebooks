import random

# little game: basically put in the state
# if you can't play little game, you are idiot
class Follower():
  def __init__(self):
    self.reset()
    
  def reset(self):
    self.n = 0
    self.state = [0,random.randint(0,1)]
    self.done = False
    return self.state

  def render(self):
    print(self.n, self.state)

  class observation_space():
    shape = (2,)

  class action_space():
    n = 2
    
  def step(self, act):
    if act == self.state[1]:
      rew = 1
    else:
      rew = 0
    self.n += 1
    if self.n == 3:
      self.done = True
    self.state = [self.n,random.randint(0,1)]
    return self.state, rew, self.done, None
  
