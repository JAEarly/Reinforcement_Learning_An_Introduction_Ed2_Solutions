from chapter02.BanditSolutions import runEGreedy, runAndPlot
from matplotlib.pyplot import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

_n = 10
_plays = 30000
_repeats = 2000

rc('text', usetex=True)
plt.rc('font', family='serif')
mpl.rcParams.update({'font.size': 14})

figure, axes = plt.subplots(figsize=(8, 8), nrows=3, ncols=1)

print('Results')
e01avg_rwds, e01avg_optimals = runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0.1, "sample_average", 0], {}, 'e=0.1', useSave=True)
e001avg_rwds, e001avg_optimals = runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0.01, "sample_average", 0], {}, 'e=0.01', useSave=True)

axes[0].set_xlabel('Plays')
axes[0].set_ylabel('Average reward')
axes[0].set_xlim(0, _plays)
axes[0].legend(loc='lower right')

axes[1].set_xlabel('Plays')
axes[1].set_ylabel('\% Optimal action')
axes[1].set_xlim(0, _plays)
axes[1].set_ylim(0, 100)
axes[1].legend(loc='lower right')

axes[2].set_xlabel('Plays')
axes[2].set_ylabel('Total reward')
axes[2].set_xlim(0, _plays)
axes[2].legend(loc='lower right')

figure, axis = plt.subplots(figsize=(8, 3), nrows=1, ncols=1)
axis.plot(np.arange(0, _plays, 1), e001avg_optimals/e01avg_optimals, color='k')
axis.plot([0, _plays], [1.088, 1.088], linestyle='--', color='k')
axis.plot([0, _plays], [1.0, 1.0], linestyle=':', color=[0.5, 0.5, 0.5])
axis.set_xlabel('Plays')
axis.set_ylabel('Optimality comparison')
axis.set_xlim(0, _plays)
axis.set_ylim(0.6, 1.2)

plt.tight_layout()
plt.show()