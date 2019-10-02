import gym
go_env = gym.make('gym_go:go-v0', size=19)
go_env.render(mode="human")
