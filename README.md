# Installation
```
pip install -e gym-go
```

# Example
```python
import gym

go_env = gym.make('gym_go:go-v0', size=7, reward_method='real')

first_action = (2,5)
second_action = (5,2)
first_state, reward, done, info = go_env.step(first_action)
go_env.render()
```

```
    0   1   2   3   4   5   6
  -----------------------------
0 |   |   |   |   |   |   |   |
  -----------------------------
1 |   |   |   |   |   |   |   |
  -----------------------------
2 |   |   |   |   |   | B |   |
  -----------------------------
3 |   |   |   |   |   |   |   |
  -----------------------------
4 |   |   |   |   |   |   |   |
  -----------------------------
5 |   |   |   |   |   |   |   |
  -----------------------------
6 |   |   |   |   |   |   |   |
  -----------------------------
	Turn: WHITE, Last Turn Passed: False, Game Over: False
	Black Area: 49, White Area: 0, Reward: 0
```

```python
second_state, reward, done, info = go_env.step(second_action)
go_env.render()
```

```
    0   1   2   3   4   5   6
  -----------------------------
0 |   |   |   |   |   |   |   |
  -----------------------------
1 |   |   |   |   |   |   |   |
  -----------------------------
2 |   |   |   |   |   | B |   |
  -----------------------------
3 |   |   |   |   |   |   |   |
  -----------------------------
4 |   |   |   |   |   |   |   |
  -----------------------------
5 |   |   | W |   |   |   |   |
  -----------------------------
6 |   |   |   |   |   |   |   |
  -----------------------------
	Turn: BLACK, Last Turn Passed: False, Game Over: False
	Black Area: 1, White Area: 1, Reward: 0
```

# Scoring
We use simple area scoring to determine the winner. A player's _area_ is defined as the number of empty points a player's pieces surround plus the number of player's pieces on the board. 
The _winner_ is the player with the larger area (a game is tied if both players have an equal amount of area on the board).

# Game Ending
A game ends when both players pass consecutively

# Reward Methods
* **Real**: 
  * `0` - Game is ongoing, white won, or game is tied
  * `1` - Black won
* **Heuristic**: `black area - white area`
