import gym
import torch

env=gym.make('CartPole-v1')
env.reset()

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
        torch.nn.Linear(6, 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)

for i in range(20):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(observation)
    print("step", i, observation, reward, done, info)

    model = TinyNet()
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.MSELoss()

    sa = torch.cat([torch.tensor(observation),torch.tensor([reward])])
    print(sa.shape)
    q = model(sa)

    # Q is the neural net. Q(
