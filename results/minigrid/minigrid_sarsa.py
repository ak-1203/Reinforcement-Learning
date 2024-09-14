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
min_epsilon=1e-2
decay_rate=0.92
Q = {((x, y),d): {z: 0 for z in range(3)} for d in range(0,4) for y in range(1, n) for x in range(1, n)}

policy={}

def epsilon_greedy(epsilon,state,policy):
    p=random.random()
    if p<=epsilon:
        action=random.randint(0,2)
    else:
        action= max(Q[state], key=Q[state].get)
    return action

def epsilon_decay(episode_num,old_epsilon):
    # new_epsilon=max(min_epsilon,(old_epsilon/episode_number))
    new_epsilon=max(min_epsilon,(old_epsilon*decay_rate))
    # new_epsilon=max(min_epsilon,old_epsilon* (decay_rate ** episode_num))

    return new_epsilon
def gen_episode(env,epsilon,state,action,policy):
    steps=0
    episode_history=[]
    while True:
        next_obs, reward, done, truncated,info = env.step(action)
        steps+=1
        next_state=(env.agent_pos,next_obs['direction'])

        next_action= epsilon_greedy(epsilon,next_state,policy)

        Q[state][action]+= alpha*(reward + (gamma*Q[next_state][next_action]) - (Q[state][action]))

        episode_history.append((state,action,next_state,reward))
        state=next_state
        action=next_action
        if done or truncated or steps>=max_steps:
            break
    return steps,episode_history
def episodeic_returns(episode):
    ep_returns=[]
    G=0
    for step in reversed(episode):
        reward=step[3]
        G = reward + gamma * G 
        ep_returns.insert(0, G)
    return ep_returns

def sarsa(env):
    epsilon=1.0
    epi_counter=0
    step_per_epi=[]
    return_per_epi=[]
    while True:
        ob=env.reset()
        s=((1,1),0)
        a=epsilon_greedy(epsilon,s,policy)
        steps,episode=gen_episode(env,epsilon,s,a,policy)
        ep_stepwise_return=episodeic_returns(episode)
        return_per_epi.append(ep_stepwise_return[0])
        step_per_epi.append(steps)
        epi_counter+=1
        updated_epsilon= epsilon_decay(epi_counter,epsilon)
        epsilon=updated_epsilon

        if epi_counter>=numOfepisodes:
            break
    return step_per_epi,return_per_epi

start_time = time.time()
step_per_epi , return_per_epi = sarsa(env)
end_time = time.time()  
execution_time = end_time - start_time
print("\n excecution time in seconds is :  ",execution_time)

print("\n final Q(s,a) : \n",Q)
for s in Q:
    policy[s] = max(Q[s], key=Q[s].get)
print("\n final policy \n",policy)

# print("\n\n\n this is steps list\n",step_per_epi)
# print("\n\n\n this is reurns list\n", return_per_epi)

epi=np.arange(1,len(step_per_epi)+1)

fig,ax =plt.subplots(2,1,figsize=(12,5),layout='constrained')
ax[0].plot(epi,return_per_epi,color='red')
ax[0].set_title("Returns Vs Episodes")
ax[0].set_xlabel("Episodes")
ax[0].set_ylabel("Returns")

ax[1].plot(epi,step_per_epi)
ax[1].set_title("No of steps Vs Episodes")
ax[1].set_xlabel("Episodes")
ax[1].set_ylabel("No of steps")
plt.suptitle("Sarsa(0) (γ=0.99 , α=0.2)")
plt.show()