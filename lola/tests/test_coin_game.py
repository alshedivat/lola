import importlib

max_steps = 1000
terminate_prob = 0.998
batch_size = 5
gameEnv = importlib.import_module('coin_game_v')
env = gameEnv.gameEnv(terminate_prob=terminate_prob, max_steps=max_steps, batch_size=batch_size)

print('state_space', env.state_space)
print('red_pos', env.red_pos)
print('blue_pos', env.blue_pos)
print('red_coin', env.red_coin)
print('coin_pos', env.coin_pos)


# test red agent picks up red coin
env.red_coin = [1, 1, 1, 1, 1]
env.red_pos = ( env.coin_pos - env.actions[1] ) % env.grid_size
state, reward, done = env.step(actions=[[1,1], [1,1], [1,1], [1,1], [1,1]])
print('red_pos', env.red_pos)
print('blue_pos', env.blue_pos)
print('red_coin', env.red_coin)
print('coin_pos', env.coin_pos)
print('reward', reward)
print('state', state)


# # test red agent picks up blue coin
# env.red_coin = 0
# env.red_pos = ( env.coin_pos - env.actions[1] ) % env.grid_size
# _, reward, done = env.step(action=1, agent='red')
# print('red_pos', env.red_pos)
# print('blue_pos', env.blue_pos)
# print('red_coin', env.red_coin)
# print('coin_pos', env.coin_pos)
# print('reward', reward)

# # test blue agent picks up red coin
# env.red_coin = 1
# env.blue_pos = ( env.coin_pos - env.actions[1] ) % env.grid_size
# _, reward, done = env.step(action=1, agent='blue')
# print('red_pos', env.red_pos)
# print('blue_pos', env.blue_pos)
# print('red_coin', env.red_coin)
# print('coin_pos', env.coin_pos)
# print('reward', reward)

# # test blue agent picks up blue coin
# env.red_coin = 0
# env.blue_pos = ( env.coin_pos - env.actions[1] ) % env.grid_size
# _, reward, done = env.step(action=1, agent='blue')
# print('red_pos', env.red_pos)
# print('blue_pos', env.blue_pos)
# print('red_coin', env.red_coin)
# print('coin_pos', env.coin_pos)
# print('reward', reward)
