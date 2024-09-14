import numpy as np
import random 
import gym
import time
import matplotlib.pyplot as plt
#--------------------------GLOBAL DECLARATION & INITIALIZING VALUES------------------------------------------------------------#
# env = gym.make("MiniGrid-Empty-6x6-v0",render_mode='human')
env = gym.make("MiniGrid-Empty-6x6-v0")

# env = gym.make("MiniGrid-Empty-16x16-v0",render_mode='human')
# env = gym.make("MiniGrid-Empty-16x16-v0")
alpha=0.2
gamma=0.99
nS=16
max_steps=100
numOfepisodes=70
min_epsilon=1e-2
decay_rate=.92
# alpha=0.02
# gamma=0.99999999
#state_representation = ((column,row),direction)
n=5#for 6x6 
#n=15 #for 16x16
# Q = {((x, y),d): {z: 0 for z in range(3)} for d in range(0,4) for y in range(1, n) for x in range(1, n)}
Q = {((x, y),d): {z: random.uniform(0, 0.1) for z in range(3)} for d in range(0,4) for y in range(1, n) for x in range(1, n)}
policy={}


#------------------------Epsilon greedy-----------------------------------------------#
def epsilon_greedy(epsilon,state,action,policy):
    p=random.random()
    if p<=epsilon:
        action=random.randint(0,2)
    else:
        # action=policy[state]
        action= max(Q[state], key=Q[state].get)
    return action

#------------------------Epsilon Decay 2 maintain balance of exploitation & exploration-------------------------------------------#

def epsilon_decay(episode_num,old_epsilon):
    # new_epsilon=max(min_epsilon,(old_epsilon/episode_num) )
    new_epsilon=max(min_epsilon,(old_epsilon*decay_rate))
    # new_epsilon=max(min_epsilon,old_epsilon* (decay_rate ** episode_num))

    return new_epsilon

#------------------------ Generating Episodes---------------------------------------------------------------------------------------#

def gen_episode(env,epsilon,state,action,policy):
    steps=0
    episode_history=[]
    while True and steps<max_steps:
        # action=random.randint(0,2)
        action=epsilon_greedy(epsilon,state,action,policy)
        steps+=1
        next_obs, reward, done, truncated,info = env.step(action)
        direction=next_obs['direction']
        next_state=(env.agent_pos,direction)
        episode_history.append(((state,action),reward,next_state,done))
        state=next_state

        if done:
            break
    return episode_history,steps

#------------------------Calculating Returns for each state of a particular episode-----------------------------------------------#
def episodeic_returns(episode):
    ep_returns=[]
    G=0
    for step in reversed(episode):
        reward=step[1]
        G = reward + gamma * G 
        ep_returns.insert(0, G)
    return ep_returns

#------------------------Main function for Monte Carlo Control--------------------------------------------------------------------#

def mc_control(env):
    
    count=0
    epsilon=1.0
    n_of_steps=[]
    epi_total_return=[]
    while True and count<numOfepisodes :
        ob=env.reset()
        state=((1,1),0)
        action=0

        
        episode,num_steps = gen_episode(env,epsilon,state,action,policy)
        n_of_steps.append(num_steps)
        count+=1
        updated_epsilon= epsilon_decay(count,epsilon)
        epsilon=updated_epsilon

        ep_stepwise_returns = episodeic_returns(episode)
        epi_total_return.append(ep_stepwise_returns[0])
        k=0
        for step in episode:
            s,a=step[0]
            Q[s][a]=Q[s][a]+alpha*(ep_stepwise_returns[k] - Q[s][a])
            k+=1
        #-----------------improving policy--------------------------------------#
        for s in Q:
            policy[s] = max(Q[s], key=Q[s].get)
    
    
    return n_of_steps,epi_total_return
# ------------Calling & others----------------------------------------
start_time = time.time()
steps_mc,mc_total_return = mc_control(env) 
end_time = time.time()  
execution_time = end_time - start_time

print("\n excecution time in seconds is :  ",execution_time)
print("\n final Q(s,a) : \n",Q)
print("\n final Policy(s) : \n",policy)
# print("this is retuens list \n \n",mc_total_return)
# print("\nlist of steps in episodes\n",steps_mc)

episode_num=np.arange(1,(len(steps_mc)+1))

fig,ax =plt.subplots(2,1,figsize=(12,5),layout='constrained')
ax[0].plot(episode_num,mc_total_return,color='red')
ax[0].set_title("Returns Vs Episodes")
ax[0].set_xlabel("Episodes")
ax[0].set_ylabel("Returns")

ax[1].plot(episode_num,steps_mc)
ax[1].set_title("No of steps Vs Episodes")
ax[1].set_xlabel("Episodes")
ax[1].set_ylabel("No of steps")
plt.suptitle("Monte Carlo(γ=0.99 , α=0.2)")
plt.show()