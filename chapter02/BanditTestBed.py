import numpy as np


class Bandit():

    def __init__(self, applyRandomWalk=False):
        self.arms = []
        self.optimalArmId = None
        self.applyRandomWalk = applyRandomWalk  # Apply random walks to arms for non-stationary bandits

    def addArm(self, arm):
        self.arms.append(arm)

    def pull(self, a):
        if not 0 <= a < self.size():
            raise IndexError("Cannot pull arm " + str(a) + " for bandit of size " + str(self.size()))
        result = self.arms[a].pull()
        if self.applyRandomWalk:
            for arm in self.arms:
                arm.walk()
        return result

    def getOptimalArmId(self):
        if self.optimalArmId is None or self.applyRandomWalk:
            bestMean = None
            for i in range(len(self.arms)):
                if bestMean is None or self.arms[i].mean > bestMean:
                    bestMean = self.arms[i].mean
                    self.optimalArmId = i
        return self.optimalArmId


    def size(self):
        return len(self.arms)


class Arm():

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def pull(self):
        return np.random.normal(self.mean, self.variance)

    def walk(self):
        self.mean += np.random.normal(0, 0.1)


def createBandit(size, nonStationary=False):
    b = Bandit(applyRandomWalk=nonStationary)
    for _ in range(size):
        mean = 0 if nonStationary else np.random.normal(0, 1)
        b.addArm(Arm(mean, 1))
    return b
