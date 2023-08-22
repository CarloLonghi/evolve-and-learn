import numpy as np
from matplotlib import pyplot as plt
import asyncio
import matplotlib.lines as mlines


async def main() -> None:

    fig, ax = plt.subplots()
    path = "./"

    trajectories = np.load(path+'trajectory.npy')
    
    ax.scatter(trajectories[0,0], trajectories[1,0], marker=",", zorder=3, color='#450053', s=30)
    ax.scatter(trajectories[0,-1], trajectories[1,-1], marker=",", zorder=3, color='#2F728F', s=30)
    targets = [(-0.2, -0.5), (0.4, -1.)]
    target1 = plt.Circle(targets[0], 0.1, color='#FDE723')
    target2 = plt.Circle(targets[1], 0.1, color='#FDE723')
    ax.add_patch(target1)
    ax.add_patch(target2)


    ax.plot(trajectories[0], trajectories[1], linewidth=1, color='#2F728F') #firebrick #6C8EBF","#9673A6"

    purple_square = mlines.Line2D([], [], color='#450053', marker='s', linestyle='None',
                            markersize=10, label='Start point')
    blue_square = mlines.Line2D([], [], color='#2F728F', marker='s', linestyle='None',
                            markersize=10, label='End point_Asexual')
    green_square = mlines.Line2D([], [], color='#62C762', marker='s', linestyle='None',
                                 markersize=10, label='End point_Sexual')
    yellow_circle = mlines.Line2D([], [], color='#FDE723', marker='o', linestyle='None',
                            markersize=10, label='Target point')
    ax.legend(handles=[purple_square, blue_square, green_square, yellow_circle])
    plt.title('Lamarckian Evolution')
    plt.figure(figsize=(3, 100))
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())







