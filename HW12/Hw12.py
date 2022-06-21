import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from PIL import Image




def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)

        self.value_layer = nn.Linear(128, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))

        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def calculateLoss(self, gamma=0.99):

        # calculating discounted rewards:
        rewards = []
        value_list = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)

        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

def train(env):
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543

    render = False
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr,betas)
    running_reward = 0
    for i_episode in range(0, 10000):
        state = env.reset()
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            if render and i_episode > 1000:
                env.render()
            if done:
                break
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()
        policy.clearMemory()

        if running_reward > 4500:
            torch.save(policy.state_dict(), 'LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            print("########## Solved! ##########")
            break
        
        if i_episode % 20 == 0:
            running_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0


def test(env, n_episodes=5, name='LunarLander_0.02_0.9_0.999.pth'):
    policy = ActorCritic()
    policy.load_state_dict(torch.load('{}'.format(name)))
    render = True
    save_gif = False
    action_list = []
    for i_episode in range(1, n_episodes+1):
        actions = []
        state = env.reset()
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        action_list.append(actions)
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

    PATH = "Action_List.npy"  # Can be modified into the name or path you want
    np.save(PATH, np.array(action_list))

if __name__ == '__main__':
    seed = 543  # Do not change this
    env = gym.make('LunarLander-v2')
    fix(env, seed)
    train(env)
    fix(env, seed)
    test(env)