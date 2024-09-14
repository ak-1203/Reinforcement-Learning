import numpy as np
import random 
import gym
import time
import matplotlib.pyplot as plt

env = gym.make("MiniGrid-Empty-6x6-v0")
n=(env.grid.width - 1)
alpha=0.2
gamma=0.99
nS=16
max_steps=100
numOfepisodes=70
min_epsilon=0.001
decay_rate=0.92

# Q = {((x, y),d): {z: random.uniform(0, 0.1) for z in range(3)} for d in range(0,4) for y in range(1, n) for x in range(1, n)}
Q = {((x, y),d): {z:0 for z in range(3)} for d in range(0,4) for y in range(1, n) for x in range(1, n)}
policy={}
def epsilon_greedy(epsilon,state):
    p=random.random()
    if p<=epsilon:
        action=random.randint(0,2)
    else:
        # action=policy[state]
        action= max(Q[state], key=Q[state].get)
    return action
def epsilon_decay(episode_num,old_epsilon):
    # new_epsilon=max(min_epsilon,(old_epsilon/epi_number) )
    # new_epsilon=max(min_epsilon,(old_epsilon*decay_rate))
    new_epsilon=max(min_epsilon,old_epsilon* (decay_rate ** episode_num))

    return new_epsilon
def gen_epi(env,state,epsilon):
    episode_history=[]
    steps=0
    while True:
        action=epsilon_greedy(epsilon,state)
        next_obs, reward, done, truncated,info = env.step(action)
        steps+=1
        
        direction=next_obs['direction']
        next_state=(env.agent_pos,direction)
        episode_history.append(((state,action),reward,next_state,done))
        
        Q[state][action]+= alpha*(reward + gamma*(max(Q[next_state].values())-Q[state][action]))
        state=next_state
        if done or steps>=max_steps or truncated:
            break
    return steps,episode_history
def episodeic_returns(episode):
    # ep_returns=[]
    G=0
    for step in reversed(episode):
        reward=step[1]
        G = reward + gamma * G 
        # ep_returns.insert(0, G)
    return G
def q_learn(env):
    epsilon=1.0
    count=0
    total_epi_return=[]
    steps_per_epi=[]
    while True:
        ob,_=env.reset()
        state=(env.agent_pos,ob['direction'])
        steps,episode= gen_epi(env,state,epsilon)
        count+=1
        next_epsilon=epsilon_decay(count,epsilon)
        epsilon=next_epsilon
        
        total_epi_return.append(episodeic_returns(episode))
        steps_per_epi.append(steps)

        for s in Q:
            policy[s] = max(Q[s], key=Q[s].get)
        if count>=numOfepisodes:
            break
    return total_epi_return,steps_per_epi

return_per_epi,step_per_epi =q_learn(env)

print(" this is Policy \n\n",policy)
print("\n this is steps list \n ",step_per_epi)
print("\n this is returns list \n ",return_per_epi)



epi=np.arange(1,len(step_per_epi)+1)
fig,ax =plt.subplots(2,1,figsize=(12,5),layout='constrained')
ax[0].plot(epi, return_per_epi,color='red')
ax[0].set_title("Returns Vs Episodes")
ax[0].set_xlabel("Episodes")
ax[0].set_ylabel("Returns")

ax[1].plot(epi, step_per_epi)
ax[1].set_title("No of steps Vs Episodes")
ax[1].set_xlabel("Episodes")
ax[1].set_ylabel("No of steps")
plt.suptitle("Q learning (γ=0.99 , α=0.08)")
plt.show()

