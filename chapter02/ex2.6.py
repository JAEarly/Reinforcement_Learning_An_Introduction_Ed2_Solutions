from chapter02.BanditSolutions import runEGreedy, runAndPlot
import matplotlib.pyplot as plt

_n = 10
_plays = 1000
_repeats = 2000

figure, axes = plt.subplots(figsize=(8, 8), nrows=3, ncols=1)

print('Results')
_, optimistic_optimality = runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0, "constant", 5], {'step_constant':0.1}, 'Optimistic', useSave=True)
runAndPlot(axes, _repeats, _plays, _n, runEGreedy, [0.1, "constant", 0], {'step_constant':0.1}, 'Realistic', useSave=True)

print(optimistic_optimality[:15])

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

plt.tight_layout()
plt.show()