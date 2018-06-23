from chapter02.BanditSolutions import runEGreedy, runAndPlot
from matplotlib.pyplot import rc
import matplotlib.pyplot as plt
import matplotlib as mpl

_n = 10
_plays = 1000
_repeats = 2000

plt.style.use('grayscale')
rc('text', usetex=True)
plt.rc('font', family='serif')
mpl.rcParams.update({'font.size': 14})

figure, axes = plt.subplots(figsize=(8, 8), nrows=3, ncols=1)
figure.patch.set_facecolor('w')

print('Results')
_, optimistic_optimality = runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0, 5, 0.1], 'Optimistic')
runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0.1, 0, 0.1], 'Realistic')

print('\nNotice spike on 10th play:')
print('Play', 'Optimality %')
for i in range(8, 25):
    print('{:^4s}'.format(str(i+1)), '{:^12s}'.format('{:^1.1f}'.format(optimistic_optimality[i])),
          'Spike' if i%10 == 0 else '')

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

plt.tight_layout()
plt.show()