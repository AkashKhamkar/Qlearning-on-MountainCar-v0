import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discerte_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.75
START_EPSILON_DECAYING = 1
END_EPSIOLON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSIOLON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discerte_os_win_size
	return(tuple(discrete_state.astype(np.int)))


for episode in range (EPISODES):
	epsiode_reward = 0
	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False
	#print("discrete_state::",q_table[discrete_state])
	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
			
		new_state, reward, done, _ = env.step(action)
		epsiode_reward += reward 
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE *( reward + DISCOUNT * max_future_q)
			q_table[discrete_state+(action, )] = new_q
		elif new_state[0] >= env.unwrapped.goal_position: # originally it is env.goal_position but gives error so this using .unwrapped
			print(f"We made it on episode {episode}")
			q_table[discrete_state + (action,)] = 0
		discrete_state = new_discrete_state

	if END_EPSIOLON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

	ep_rewards.append(epsiode_reward)

	if episode % 100 == 0:
		np.save(f"qtables/{episode}-qtable.npy", q_table)

	if not episode % SHOW_EVERY:
		average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

		print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")



env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=2)
plt.grid(True)
plt.show()