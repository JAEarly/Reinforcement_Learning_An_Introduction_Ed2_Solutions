from chapter02.BanditSolutions import runEGreedy, runAndPlot
import matplotlib.pyplot as plt
import numpy as np

_n = 10
_plays = 20000
_repeats = 2000

figure, axes = plt.subplots(figsize=(8, 8), nrows=3, ncols=1)

print('Results')
e01avg, e01rwd = runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0.1, "sample_average", 0], {}, 'e=0.1', useSave=True)
e001avg, e001rwd = runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0.01, "sample_average", 0], {}, 'e=0.01', useSave=True)
#runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0, "sample_average", 0], {}, 'e=0', useSave=True)

axes[0].set_xlabel('Plays')
axes[0].set_ylabel('Average reward')
axes[0].set_xlim(0, _plays)
axes[0].legend(loc='lower right')

axes[1].set_xlabel('Plays')
axes[1].set_ylabel('% Optimal action')
axes[1].set_xlim(0, _plays)
axes[1].set_ylim(0, 100)
axes[1].legend(loc='lower right')

axes[2].set_xlabel('Plays')
axes[2].set_ylabel('Total reward')
axes[2].set_xlim(0, _plays)
axes[2].legend(loc='lower right')

figure, axis = plt.subplots(figsize=(8, 2.6), nrows=1, ncols=1)
axis.plot(np.arange(0, _plays, 1), np.cumsum(e001avg)/np.cumsum(e01avg))
axis.plot([0, _plays], [1.088, 1.088])
axis.set_xlabel('Plays')
axis.set_ylabel('e=0.01 vs e=0.1')
axis.set_xlim(0, _plays)
axis.set_ylim(0.9, 1.1)

plt.tight_layout()
plt.show()