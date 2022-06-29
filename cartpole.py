import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
env=gym.make('CartPole-v1')
env.reset()
MODEL_PATH = 'saved_model.pt'
class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
        torch.nn.Linear(6, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
        )
    def forward(self, x):
        return self.net(x)
"""
q(s, a) is a neural net. We randomly sample actions, and use the reward as the groundtruth for the model.
The model here acts kind of like a monte carlo algorithm, where it learns from repeated trials
an epoch is a single simulation
AVERAGE STEPS BEFORE FAIL:
random sampling: 21.4125
naive ai: 
"""
model=TinyNet()
loss = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-6)
tot = 0
prev_r = 0
prev_sa = 0
for i in tqdm(range(1000)):
    for j in range(1000):
        random_action=env.action_space.sample()
        s=F.one_hot(torch.tensor(random_action), num_classes=2)
        observation, reward, done, info = env.step(random_action)
        if done == True:
            env.reset()
            tot += j
            break
        sa=torch.cat([torch.tensor(observation),s])
        if prev_r != 0:
            # q_new = ((1-alpha) * net(prev_sa)) + alpha * (reward + argmax(q(
            loss.zero_grad()
            q=model(prev_sa)
            l = loss(torch.tensor(reward), q)
            l.backward()
            optim.step()
            print(f'step {i} obs {observation} reward {reward} done? {done} info {info} loss: {l.item()}')
        prev_r = reward
        prev_sa = sa
tot /= 1000
p_tot = tot
print('sampling from neural net...')
tot = 0
prev_obs = None
# this time, we randomly sample from the neural net
for i in tqdm(range(1000)):
    for j in range(1000):
        # which is better?
        a0 = torch.tensor([0, 1])
        a1 = torch.tensor([1, 0])
        best_q = 0
        if prev_obs != None:
            q0 = model(torch.cat([prev_obs, a0]))
            q1 = model(torch.cat([prev_obs, a1]))
            if q0 > q1:
                best_q = 0
            else:
                best_q = 1
        else:
            best_q = env.action_space.sample()
        print(best_q)
        observation, reward, done, info = env.step(best_q)
        print(f'step {i} obs {observation} reward {reward} done? {done} info {info} loss: {l.item()}')
        prev_obs = torch.tensor(observation)
        if done == True:
            env.reset()
            tot += j
            break
print(tot / 1000)
print(p_tot)
