import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from typing import Tuple, List, Union


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def bstack(bb: List[Union[float, np.ndarray]]) -> List[np.ndarray]:
    # reduced loop version of bstak
    l, ll = len(bb), len(bb[0])
    return [np.array([i[j] for i in bb]).reshape(l, -1) for j in range(ll)]


def to_one_hot(a: np.ndarray, K: int, a_dim: int) -> np.ndarray:
    # vectorized version of to_one_hot
    one_hot_action = np.zeros((K * a_dim))
    index = np.arange(0, K * a_dim, a_dim)
    index = index[a >= 0]
    one_hot_action[a[a >= 0] + index] = 1
    return np.split(one_hot_action, K)


def reformat_batch(batch: np.ndarray, K: int, a_dim: int, remove_policy=False) -> Tuple[List[np.ndarray]]:
    X, Y = [], []
    for o, a, outs in batch:
        a = np.array(a)
        x = [o] + to_one_hot(a, K, a_dim)

        # flatten outs
        y = [item for sublist in outs for item in sublist]

        X.append(x)
        Y.append(y)

    X, Y = bstack(X), bstack(Y)

    if remove_policy:
        nY = [Y[0]]
        for i in range(3, len(Y), 3):
            nY.append(Y[i])
            nY.append(Y[i + 1])
        Y = nY
    else:
        Y.pop(1)

    return X, Y


class DenseRepresentation(nn.Module):
    # h network
    def __init__(self, o_dim: int, s_dim: int, hidden_layer_dim: int, hidden_layer_count: int) -> None:
        super().__init__()

        sequential = [nn.Linear(o_dim, hidden_layer_dim), nn.ELU()]
        sequential += [nn.Linear(hidden_layer_dim,
                                 hidden_layer_dim), nn.ELU()] * hidden_layer_count
        self.out = nn.Linear(hidden_layer_dim, s_dim)

        self.linearReluStack = nn.Sequential(*tuple(sequential))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.linearReluStack(state)
        return self.out(x)


class DenseDynamics(nn.Module):
    # g network
    def __init__(self, s_dim: int, a_dim: int, hidden_layer_dim: int, hidden_layer_count: int) -> None:
        super().__init__()

        sequential = [nn.Linear(s_dim + a_dim, hidden_layer_dim), nn.ELU()]
        sequential += [nn.Linear(hidden_layer_dim,
                                 hidden_layer_dim), nn.ELU()] * hidden_layer_count

        self.linearReluStack = nn.Sequential(*tuple(sequential))
        self.out1 = nn.Linear(hidden_layer_dim, 1)
        self.out2 = nn.Linear(hidden_layer_dim, s_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state.T, action.T]).T
        x = self.linearReluStack(x)

        return self.out1(x), self.out2(x)


class DensePrediction(nn.Module):
    # f network
    def __init__(self, s_dim: int, a_dim: int, hidden_layer_dim: int, hidden_layer_count: int, with_policy: bool = True) -> None:
        super().__init__()
        self.with_policy = with_policy

        sequential = [nn.Linear(s_dim, hidden_layer_dim), nn.ELU()]
        sequential += [nn.Linear(hidden_layer_dim,
                                 hidden_layer_dim), nn.ELU()] * hidden_layer_count

        self.linearReluStack = nn.Sequential(*tuple(sequential))
        self.out1 = nn.Linear(hidden_layer_dim, a_dim)
        self.out2 = nn.Linear(hidden_layer_dim, 1)

    def forward(self, state: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = self.linearReluStack(state)

        if self.with_policy:
            return self.out1(x), self.out2(x)
        return self.out2(x)


class MuModel:
    LAYER_COUNT = 4
    LAYER_DIM = 128
    BN = False

    def __init__(self, observation_dim: int, action_dim: int, s_dim: int = 8, K: int = 5, lr: float = 1e-3, with_policy: bool = True, device='cpu') -> None:
        self.observation_dim, self.action_dim, self.s_dim = observation_dim, action_dim, s_dim
        self.K, self.lr, self.with_policy = K, lr, with_policy
        self.device = device

        self.h = DenseRepresentation(
            o_dim=observation_dim[0], s_dim=s_dim, hidden_layer_dim=self.LAYER_DIM, hidden_layer_count=self.LAYER_COUNT).to(device)
        self.g = DenseDynamics(s_dim=s_dim, a_dim=action_dim, hidden_layer_dim=self.LAYER_DIM,
                               hidden_layer_count=self.LAYER_COUNT).to(device)
        self.f = DensePrediction(s_dim=s_dim, a_dim=action_dim, hidden_layer_dim=self.LAYER_DIM,
                                 hidden_layer_count=self.LAYER_COUNT, with_policy=with_policy).to(device)

        params = list(self.h.parameters()) + \
            list(self.g.parameters()) + list(self.f.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.losses = []

        # make class compatible with Geohot's other code
        self.o_dim = self.observation_dim
        self.a_dim = self.action_dim
        Mu = namedtuple('mu', 'predict')
        self.mu = Mu(self.predict)

    def forward(self, X: List[torch.Tensor], train: bool = True) -> List[torch.Tensor]:
        self.h.eval(), self.g.eval(), self.f.eval()
        if train:
            self.h.train(), self.g.train(), self.f.train()

        X = [torch.from_numpy(x.astype(np.float32)).to(self.device) for x in X]
        Y_pred = []

        state = self.h(X[0])
        if self.with_policy:
            policy, value = self.f(state)
            Y_pred += [value, policy]
        else:
            value = self.f(state)
            Y_pred.append(value)

        for k in range(self.K):
            reward, new_state = self.g(state, X[k + 1])
            if self.with_policy:
                policy, value = self.f(state)
                Y_pred += [value, reward, policy]
            else:
                value = self.f(state)
                Y_pred += [value, reward]

            state = new_state

        return Y_pred

    def predict(self, X: List[torch.Tensor]) -> List[torch.Tensor]:
        with torch.no_grad():
            Y_pred = self.forward(X, train=False)
        return Y_pred

    def train(self, batch: List[np.ndarray]) -> None:
        self.h.train(), self.g.train(), self.f.train()
        losses = []
        mse, smcel = F.mse_loss, nn.BCEWithLogitsLoss()

        X, Y = reformat_batch(
            batch, self.K, self.action_dim, not self.with_policy)
        Y = [torch.from_numpy(y.astype(np.float32)).to(self.device) for y in Y]
        Y_pred = self.forward(X, train=True)

        losses.append(mse(Y_pred[0], Y[0]))
        if self.with_policy:
            losses.append(smcel(Y_pred[1], Y[1]))

        for k in range(self.K):
            losses.append(mse(Y_pred[3 * k + 2], Y[3 * k + 2]))
            losses.append(mse(Y_pred[3 * k + 3], Y[3 * k + 3]))
            if self.with_policy:
                losses.append(smcel(Y_pred[3 * k + 4], Y[3 * k + 4]))

        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append([loss.item()] + [l.item() for l in losses])

    def ht(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.from_numpy(
                    state.astype(np.float32)).to(self.device)
            return self.h(state)

    def ft(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.from_numpy(
                    state.astype(np.float32)).to(self.device)
            if self.with_policy:
                policy, value = self.f(state)
                return policy.exp(), value
            else:
                value = self.f(state)
                return value
