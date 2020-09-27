import argparse

import gym

# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--boardsize', type=int, default=7)
parser.add_argument('--komi', type=float, default=0)
args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi)

# Game loop
done = False
while not done:
    go_env.render(mode="terminal")
    move = input("Input move '(row col)/p': ")
    if move == 'p':
        action = None
    else:
        action = int(move[0]), int(move[2])
    state, reward, done, info = go_env.step(action)

    if go_env.game_ended():
        break
    action = go_env.uniform_random_action()
    state, reward, done, info = go_env.step(action)
go_env.render(mode="terminal")
