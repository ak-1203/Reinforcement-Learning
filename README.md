# Title : Reinforcement-Learning

## Description:
This file consists of tasks I have done and learnings I got while studying RL.
This file includes the Explanation of enviornments used in each task , Graphical results after implementing various algorithms and Challenges I faced while implementation .

## Task  : Solving Frozen Lake Environment using Dynamic Programming  .
  
### Documentation : https://www.gymlibrary.dev/environments/toy_text/frozen_lake/   
### Detailed Description About Frozen Lake Environment:  
  
### Observation Space 
Let us cosider 4x4 map for the discussion purpose  
env.observation_space= Discrete(16)  
Here each of the state is represented by a whole number ranging [0,15]  
Further custom map can be defined and  each state can be labelled as either S (start) or F(Frozen) or H(Hole) or G(Goal)  
By default mapping , S is at 0 and G is at 15.  
> In General dor nxn : state_representation = current_row * nrows + current_col ( rows and columns start from zero )

### Action Space  
> There are 4 possible action that an agent can take :  
       * 0 : Left
       * 1 : Down
       * 0 : Right
       * 0 : Up
       
### Rewards
       * 1 , reaching Goal
       * 0 , otherwise

## Task : Implementation of Model Free Control Algorithms in Minigrid Empty Space Enviornment.

### Documentation : https://minigrid.farama.org/environments/minigrid/EmptyEnv/
### Detailed Description About Minigrid Environment:

### Observation Space : 
env.observation_space = { 'image': Box(0,255,(7,7,3),uint8) , 'direction':Discrete(4) , 'mission' : "Get to the green square" }  
Significance of key image :  
It basically represents the information that the agent can perceives at any grid cell.  
* The (7,7,3) matrix : at any grid cell the agent can be visualized at the center of the 7x7 grid ( any of those 7x7 grid cells may be behind the walls i.e can be outside environment) , these 7x7 cells can be termed as 'Vision of Agent' (at any given state , number of states about which the agent holds information). 
       * The 3 in (7x7x3) is for the color intensity of a particular grid cell ( among that 7x7) There are 3 elements in the matrix R,G,B each containing intensity in the range [0,255].  

* Significance of key direction:  
Value of this can be any number in the interval [0,3] which basically tend to denote the direction the agent is facing at any state.  
       * 0 : Right
       * 1 : Down
       * 2 : Left
       * 3 : Up

* Significance of key Mission:    
It contains a string value which represents the goal of the agent.  
### Action Space : 
Relevant actions for Empty Space Environment are :  0,1,2  
* 0 : Turn Left   
* 1 : Turn Right  
* 2 : Move Forward  
> When we take any action a:  
env.step(a) = ( next_obs , reward , False/boolean ,  False/boolean , { } )  
Here , next_obs is observaion space of the next state after taking action a   
       reward is the immediate reward received after taking action a   
       Third element of tuple is a boolean expression which is True only when the agent reaches goal , it is called termination.  
       Fourth element of tuple is a boolean expression which is True only when the episode terminates due to the max_steps condition , it is called truncation.  
       Last element of tuple incude the extra information regarding the state.


## Steps per episode and Returns per episode graphs :
### For monte carlo:
<img width="650"  alt="image" src="results/minigrid/mc.png">

### For Sarsa:
<img width="650"  alt="image" src="results/minigrid/sarsa_0.png">

### For Sarsa(λ):
<img width="650"  alt="image" src="results/minigrid/sarsaλ_2.png">

### For Q-learning:
<img width="650"  alt="image" src="results/minigrid/q_learn.png">


    


