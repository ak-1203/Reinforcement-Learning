# Title : Reinforcement-Learning

## Description:
This file consists of knowledge I gained in the recent months in Learinng about RL and about tasks that was implemented in the learning phase.
I took a deep dive in learning about RL , Markov Descision Process , Dynamic Programming , Model free prediction and Model free control.
Challenges faced was mostly in understanding the concepts and visualizing them with real time examples

## Task  : Implementation of Value Iteration and Policy Iteration in Frozen Lake Environment with completely defined model.



## Task : Implementation of Model Free Control Algorithms in Minigrid Empty Space Enviornment.

### Documentation : https://minigrid.farama.org/environments/minigrid/EmptyEnv/
### Detailed Description About Minigrid Environment:

### Observation Space : 
env.observation_space = { 'image': Box(0,255,(7,7,3),uint8) , 'direction':Discrete(4) , 'mission' : "Get to the green square" }  
Significance of key image :  
It basically represents the information that the agent can perceives at any grid cell.  
* The (7,7,3) matrix : at any grid cell the agent is at the center of the 7x7 grid cell ( any of those 7x7 grid cells may be behind the walls i.e can be outside environment). 
       * The 3 in (7x7x3) is for the color intensity of a particular grid cell ( among that 7x7) There are 3 elements in the matrix R,G,B each containing number between 0 and 255.  

* Significance of key direction:  
Value of this can be any number in the interval [0,3] which basically tend to denote the direction the agent is facing at any state.  

* Significance of key Mission:    
It contains a string value which represents the goal of the agent.  
### Action Space : 
Relevant actions for Empty Space Environment are :  0,1,2  
* 0 : Left   
* 1 : Right  
* 2 : Forward  
> When we take any action a:  
env.step(a) = ( next_obs , reward , False/boolean ,  False/boolean , { } )  
Here ,  next_obs is observaion space of the next state after taking action a   
       reward is the immediate reward received after taking action a   
       Third element of tuple is a boolean expression which is True only when the agent reaches goal , it is called termination.  
       Fourth element of tuple is a boolean expression which is True only when the episode terminates due to the max_steps condition , it is called truncation.  
       Last element of tuple incude the extra information regarding the state.


## Steps per episode and Returns per episode graphs :
### For monte carlo:
<img width="264" alt="image" src="https://user-images.githubusercontent.com/20359930/146223524-e07f7dd8-7e5e-40e2-a374-fdb20f987153.png">




    


