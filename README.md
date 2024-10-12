# Reinforcement Learning

## Overview

This repository documents various tasks and insights gained through the study of Reinforcement Learning (RL). It includes detailed explanations of the environments used, graphical results from the implementation of different algorithms, and an overview of the challenges encountered during the process.

## Solving Frozen Lake Environment using Dynamic Programming

### Documentation:  
[Frozen Lake Environment Documentation](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)  

Frozen lake involves crossing a frozen lake from Start(S) to Goal(G) without falling into any Holes(H) by walking over the Frozen(F) lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake. 

### Frozen Lake Environment Overview

#### Observation Space:  

Consider a 4x4 grid for discussion purpose:  

<img src="https://github.com/user-attachments/assets/a8b10a1c-c9c2-4eac-9825-89a005c17324" alt="image" width="300" height="300">  

  
- **Observation space = Discrete(16)**  
- Each state is represented by a number ranging from 0 to 15.  
- A custom map can be defined, with each state labeled as S (Start), F (Frozen), H (Hole), or G (Goal).  
- In the default map, S is at 0 and G is at 15.  

**For a general n x n grid:**  
State representation = `current_row * nrows + current_col`  
(Where rows and columns start from zero)

#### Action Space:  
The agent can take one of four possible actions:  
<!--- **0:** Left  
- **1:** Down  
- **2:** Right  
- **3:** Up  -->
| Value | Action     |
|-------|----------- |
| 0     | Turn Left  |
| 1     | Turn Down  |
| 2     | Turn Right |
| 3     | Turn Up    |  

#### Transition Matrix , P :
For each pair of states in `[1, nS]` and actions in `[1, nA]`, `P[state][action]` is a tuple in the form of `(probability, next_state, reward, terminal)`.

- **Non-Slippery Nature**: This represents deterministic behavior of the environment, meaning the transition probabilities for any state-action pair are always `1`.
  
- **Slippery Nature**: This represents stochastic behavior of the environment, where, for any given state and action, the transition probability is typically `1/3` (or `0.33`). Due to this, the agent may not always move in the intended direction.

#### Rewards:  
- **1 :**  On Reaching the goal  
- **0 :**  Otherwise  

### Frozen Lake Result:  
![PI and VI result](results/frozen_lake/result-gif.gif)  

---

## Implementation of Model-Free Control Algorithms in the Minigrid Empty Space Environment

### Documentation:  
[Minigrid Empty Space Environment Documentation](https://minigrid.farama.org/environments/minigrid/EmptyEnv/)  
Minigrid Empty Space Environment, as the name suggests, contains no obstacles between the start and the goal, making it an ideal setting for validating various Reinforcement Learning (RL) algorithms. The objective is to navigate the agent to the green square, where a sparse reward is provided upon successful completion.  

### Minigrid Environment Overview

#### Observation Space:  
<img src="https://github.com/user-attachments/assets/1bfd2221-4bf6-4d6f-9586-28ef213b07e0" alt="image" width="300" height="300">

- **env.observation_space = { 'image': Box(0, 255, (7, 7, 3), uint8), 'direction': Discrete(4), 'mission': "Get to the green square" }**  

**Key Elements:**
- **Image:**  
  Represents the agent's view at any grid cell as a (7, 7, 3) matrix.  
  - The agent is visualized at the center of the 7x7 grid, can be termed as its "vision" (the set of grid cells it can perceive).  
  - The 3 channels in the matrix represent the RGB color intensities of each grid cell, ranging from 0 to 255.

- **Direction:**  
 Denotes the direction the agent is facing, with values in the range [0, 3]:
    
  | Value | Direction Faced |
  |-------|-----------|
  | 0     | Right     |
  | 1     | Down      |
  | 2     | Left      |
  | 3     | Up        |

- **Mission:**  
  A string that defines the agent's goal, typically "Get to the green square."

#### Action Space:  
Relevant actions for this environment:  

| Value | Action     |
|-------|----------- |
| 0     | Turn Left  |
| 1     | Turn Down  |
| 2     | Turn Right | 

When an action `a` is taken:  
`env.step(a)` returns a tuple:  
- `next_obs`: The observation space after taking action `a`.  
- `reward`: The immediate reward received after the action.  
- `done`: Boolean, `True` if the agent reaches the goal (termination condition).  
- `truncated`: Boolean, `True` if the episode terminates due to exceeding `max_steps`.  
- `info`: Additional information about the state.
#### Reward Function : 
- **Success:** A reward is calculated using the formula:  
  `Reward = 1 - 0.9 * (step_count / max_steps)`

- **Failure:** A reward of `0` is given.

---

### Minigrid Results:

### Monte Carlo:
![Monte Carlo Results](results/minigrid/mc.png)

### Sarsa:
![Sarsa Results](results/minigrid/sarsa_0.png)

### Sarsa(λ):
![Sarsa(λ) Results](results/minigrid/sarsaλ_2.png)

### Q-Learning:
![Q-Learning Results](results/minigrid/q_learn.png)

