import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('ppo_model_states/fitnesses.csv')

plt.plot(data['fitness'])
plt.show()