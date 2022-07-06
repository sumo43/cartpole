import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import wandb

"""
q(s, a) is a neural net. We randomly sample actions, and use the reward as the groundtruth for the model.
The model here acts kind of like a monte carlo algorithm, where it learns from repeated trials
an epoch is a single simulation
AVERAGE STEPS BEFORE FAIL:
random sampling: 21.4125
ai: 60?
"""

env=gym.make('CartPole-v1')
batch_size = 2

gamma=0.95

class Trainer:
    memory = []
    def remember(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def experience_replay(self, model, optimizer, loss):
        if len(self.memory) < batch_size:
            return
        for step in random.sample(self.memory, 1):
            state, action, next_state, reward = step
            terminal = reward < 1

            optimizer.zero_grad()
            if terminal:
                q = model(torch.tensor(state))[action]
                l = loss(q, torch.tensor(reward))
            else:
                q = model(torch.tensor(state))[action]
                next_q = torch.amax(model(torch.tensor(next_state)))
                l = loss(q, reward + gamma * next_q)
            l.backward()
            optimizer.step()
    
    def reset_memory(self):
        self.memory = []

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)


def main():

    NUM_EPOCHS = 100
    NUM_EPISODES = 100
    epsilon = 0.1 # propensity to explore

    wandb.init()
    
    loss = torch.nn.MSELoss()
    model=TinyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    trainer = Trainer()
    memory = []
    wandb.watch(model)

    for i in range(NUM_EPOCHS):
        for j in tqdm(range(NUM_EPISODES)):
            k = 0
            state = env.reset()
            while True:
                k += 1
                ex2 = random.uniform(0,1)
                if ex2 > epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(torch.argmax(model(torch.tensor(state))))

                next_state, reward, done, info = env.step(action)
                if done:
                    if k >= 200:
                        reward = 5
                    else:
                        reward = -1.

                trainer.remember(state, action, next_state, reward)
                trainer.experience_replay(model, optimizer, loss)
                if done:
                    break

        test_model(model)
    
def test_model(model):
    print("Testing model...")
    fails=0
    for i in tqdm(range(100)):
        state = env.reset()
        j = 0
        while True:
            j += 1
            action = torch.argmax(model(torch.tensor(state)))
            state, reward, done, info = env.step(int(action))
            if done:
                print(j) 
                torch.save(model.state_dict(), 'saved_model.pt')
                fails += j
                j = 0
                break

    print(f'avg: {fails/100}')

    if(fails/100 >= 140):
        torch.save(model.state_dict(), 'model.pt')

    

if __name__ == "__main__":
    main()