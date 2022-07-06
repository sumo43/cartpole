import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

"""
q(s, a) is a neural net. We randomly sample actions, and use the reward as the groundtruth for the model.
The model here acts kind of like a monte carlo algorithm, where it learns from repeated trials
an epoch is a single simulation
AVERAGE STEPS BEFORE FAIL:
random sampling: 21.4125
ai: 60?
"""

batch_size = 4
gamma=0.95
env=gym.make('CartPole-v1')

class Trainer:
    memory = []
    def remember(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def experience_replay(self, model, optimizer, loss):
        if len(self.memory) < batch_size:
            return
        for step in random.sample(self.memory, batch_size):
            state, action, next_state, reward = step
            terminal = reward < 1

            loss.zero_grad()
            if terminal:
                l = loss(q, reward)
            else:
                q = model(torch.tensor(state))[action]
                next_q = torch.amax(model(torch.tensor(next_state)))
                l = loss(q, reward + gamma * next_q)
            l.backward()
            optimizer.step()

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
        torch.nn.Linear(4, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 2)
        )
    def forward(self, x):
        return self.net(x)

def main():

    NUM_EPISODES = 1000
    gamma=.95

    model=TinyNet()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer()
    memory = []

    for i in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        while True:
            loss.zero_grad()
            random_action = env.action_space.sample()
            next_state, reward, done, info = env.step(random_action)
            trainer.remember(state, random_action, next_state, reward)
            trainer.experience_replay(model, optimizer, loss)
            if done:
                break

    test_model(model)

def test_model(model):

    print("Testing model...")
    fails=0
    for i in tqdm(range(200)):
        state = env.reset()
        j = 0
        while True:
            j += 1
            action = torch.argmax(model(torch.tensor(state)))
            state, reward, done, info = env.step(int(action))
            if done:
                print(j)
                fails += j
                j = 0
                break

    print(fails / 200)

    

if __name__ == "__main__":
    main()