# evolve-and-learn

To run experiments for DRL-PPO:
1. Run python DRL/PPO/optimize.py
2. Add --visualize to show the simulation
3. Add --from_checkpoint to restart from the last optimization checkpoint

To plot rewards and state values run python DRL/PPO/plot_statistics.py

To see the last optimized controller run python DRL/PPO/rerun_best.py

If you want to optimize another morphology, edit the DRL/PPO/optimize.py file
