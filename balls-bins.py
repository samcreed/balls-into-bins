import numpy as np
import matplotlib.pyplot as plt

# In this file, utility functions for generating balls and bins experiments.
# Given a list of bins, should be able to randomly pick one and place a ball into it.


PENALTY = True

########################################################################################################
# Simple structure for a ball. May have weight (job size). Possible TODO: multi-dimensional weight?
class Ball():
	def __init__(self, weight):
		self.weight = weight

########################################################################################################
# Simple structure for a bin. May have maximum capacity (processing power).
# TODO: could try and model submodular load function on bins (is that a thing? More jobs = slower overall)
class Bin():
	def __init__(self, capacity):
		self.load = 0
		self.capacity = capacity
		self.items = []

	def add(self, ball):
		retval = (self.load + ball.weight <= self.capacity)
		if retval:
			if PENALTY:
				self.load += ball.weight + 0.5 * self.load # heavy penalty for choosing loaded target
			else:
				self.load += ball.weight
			self.items.append(ball)
		return retval

########################################################################################################
# Simple structure for a collection of bins. Generates new bins based on given distribution.
class Bins():
	def __init__(self, settings):
		self.settings = settings
		self.bins = np.empty(self.settings.bins, dtype=object)
		for i in range(self.settings.bins):
			self.bins[i] = Bin(self.settings.capacity)

	def __len__(self):
		return len(self.bins)


	def __getitem__(self, item):
		return self.bins[item]

		# TODO: generate bins by distribution?

########################################################################################################
# Basic parameter settings for an experiment.
class Settings():
	def __init__(self, balls=1000, bins=1000, ballWMean=1, ballWStdD=0, binWMean=1, binWStdD=0, graphStruct=None, capacity=1000):
		self.balls = balls
		self.bins = bins
		self.ballWMean = ballWMean
		self.ballWStdD = ballWStdD
		self.binWMean = binWMean
		self.binWStdD = binWStdD
		self.graphStruct = graphStruct
		self.capacity = capacity

	def newBall(self):
		# TODO: if there is some non-uniform ball weight, generate here with standard deviation and distribution
		return Ball(self.ballWMean)


########################################################################################################
# Super-class for different types of ball-allocation strategies we might want to try.
class Strategy():
	def __init__(self):
		self.a = 0

	# Abstract method, to be filled in by subclass strategies
	def placeBall(self, bins, ball):
		return []

########################################################################################################
# TODO: implement various ball allocation strategies.

# Place ball in best of D random choices.
class Greedy(Strategy):
	def __init__(self, d=2):
		self.d = d
		self.a = 0

	def placeBall(self, bins, ball):
		# Randomly choose D bin candidates in which to place a ball.
		choices = []
		for i in range(self.d):
			self.a += 1
			bin = np.random.randint(len(bins))
			choices.append(bin)

		# Place ball in selected bin that is least loaded.
		minBin = choices[0]
		for bin in choices:
			if bins[bin].load < bins[minBin].load:
				minBin = bin

		bins[minBin].add(ball)
		return bins

# Paritition bins into d sets, uniformly sample them.
class Left(Strategy):
	def __init__(self, d=2):
		self.d = d
		self.a = 0

	def placeBall(self, bins, ball):
		# Randomly choose D bin candidates in which to place a ball.
		choices = []
		for i in range(self.d):
			self.a += 1
			bin = np.random.randint(len(bins)/self.d)
			choices.append((i+1)*bin)

		# Place ball in selected bin that is least loaded.
		minBin = choices[0]
		for bin in choices:
			if bins[bin].load < bins[minBin].load:
				minBin = bin

		bins[minBin].add(ball)
		return bins

# nearly optimal strat; each ball must know its placement/order in sequence.
# what happens if we allow some leeway in it's correctness?
class Adaptive(Strategy):
	def __init__(self):
		self.i = 0.0
		self.a = 0

	def placeBall(self, bins, ball):
		placed = False
		while not placed:
			self.a += 1
			minBin = np.random.randint(len(bins))
			if bins[minBin].load < (self.i/len(bins)) + 1:
				placed = True
		
		bins[minBin].add(ball)
		self.i += 1
		
		return bins

# Start at some random point, roll down in d-dimension gravity well.
class Grid(Strategy):
	def __init__(self, d=1):
		self.d = d
		self.a = 0

	def placeBall(self, bins, ball):
		# Randomly choose starting point
		minBin = np.random.randint(len(bins))

		localMin = False
		while not localMin:
			choices = []
			# construct grid shape for local search
			for i in range(self.d):
				self.a += 1
				delta = int(np.power(np.power(len(bins), 1.0/self.d), i))
				choices.append((minBin - delta) % len(bins))
				choices.append((minBin + delta) % len(bins))

			minDir = choices[0]
			for choice in choices:
				if bins[choice].load < bins[minDir].load:
					minDir = choice

			if bins[minBin].load <= bins[minDir].load:
				localMin = True
			else:
				minBin = minDir

		bins[minBin].add(ball)
		return bins

########################################################################################################
# Basic structure for Experiment results.
# Calculates difference between minimum and maximum filled bins + mean, standard deviation.
class Results():
	def __init__(self, name):
		self.name = name
		self.n = 0
		self.a = 0
		self.maxLoad = 0
		self.avgLoad = 0
		self.minLoad = 0

	def record(self, bins, a):
		# Record the given bin allocation, determine difference between max and average load.
		self.n = len(bins)
		self.a = a
		self.maxLoad = max([bin.load for bin in bins])
		self.avgLoad = sum([bin.load for bin in bins]) / len(bins)
		self.minLoad = min([bin.load for bin in bins])

########################################################################################################
# Every experiment has some parameter settings, ball placement strategy, and some number of iterations to run for.
# To use: run() performs the experiment for the given number of iterations. Results returns the results.
class Experiment():
	def __init__(self, name, settings, strategy):
		self.name = name
		self.settings = settings
		self.strategy = strategy
		self.results = Results(name)
		self.bins = Bins(settings)

	def run(self):
		for i in range(self.settings.balls):
			ball = self.settings.newBall()
			self.bins = self.strategy.placeBall(self.bins, ball)
		self.results.record(self.bins, self.strategy.a)

	def getResults(self):
		return self.results

#########################################################################################################
# compare all results... for growing n, record difference for varying n
def displayResults(results, trials):

	greedyD1load = []
	greedyD2load = []
	greedyD1a = []
	greedyD2a = []
	leftD2load = []
	leftD2a = []
	gridD1load = []
	gridD2load = []
	gridD3load = []
	gridD1a = []
	gridD2a = []
	gridD3a = []
	adaptiveload = []
	adaptivea = []

	for result in results:
		if result.name == "uniform D=1 greedy":
			greedyD1load.append(result.maxLoad)
			greedyD1a.append(result.a)
		elif result.name == "uniform D=2 greedy":
			greedyD2load.append(result.maxLoad)
			greedyD2a.append(result.a)
		elif result.name == "uniform D=2 left":
			leftD2load.append(result.maxLoad)
			leftD2a.append(result.a)
		elif result.name == "uniform D=1 grid":
			gridD1load.append(result.maxLoad)
			gridD1a.append(result.a)
		elif result.name == "uniform D=2 grid":
			gridD2load.append(result.maxLoad)
			gridD2a.append(result.a)
		elif result.name == "uniform D=3 grid":
			gridD3load.append(result.maxLoad)
			gridD3a.append(result.a)
		elif result.name == "uniform adaptive":
			adaptiveload.append(result.maxLoad)
			adaptivea.append(result.a)

	greedyD1load.sort()
	greedyD2load.sort()
	greedyD1a.sort()
	greedyD2a.sort()
	leftD2load.sort()
	leftD2a.sort()
	gridD1load.sort()
	gridD2load.sort()
	gridD3load.sort()
	gridD1a.sort()
	gridD2a.sort()
	gridD3a.sort()
	adaptiveload.sort()
	adaptivea.sort()

	x = xrange(trials)

	plt.plot(x, greedyD1load, label='uniform D=1 choices')
	plt.plot(x, greedyD2load, label='uniform D=2 choices')
	plt.title('Power of Two Choices')
	plt.legend(loc='upper left')
	plt.xlabel('Trials')
	plt.ylabel('Max Load')
	plt.show()
	
	plt.plot(x, greedyD2load, label='uniform D=2 greedy')
	plt.plot(x, leftD2load, label='uniform D=2 left')
	plt.title('Comparing Greedy and Left')
	plt.legend(loc='upper left')
	plt.xlabel('Trials')
	plt.ylabel('Max Load')
	plt.show()

	plt.plot(x, gridD1load, label='uniform D=1 grid')
	plt.plot(x, gridD2load, label='uniform D=2 grid')
	plt.plot(x, gridD3load, label='uniform D=3 grid')
	plt.title('Comparing Grid Dimension for Max Load')
	plt.legend(loc='upper left')
	plt.xlabel('Trials')
	plt.ylabel('Max Load')
	plt.show()

	plt.plot(x, gridD1a, label='uniform D=1 grid local search')
	plt.plot(x, gridD2a, label='uniform D=2 grid local search')
	plt.plot(x, gridD3a, label='uniform D=3 grid local search')
	plt.title('Comparing Grid Dimension for Allocation Time')
	plt.legend(loc='upper left')
	plt.xlabel('Trials')
	plt.ylabel('Allocation Time')
	plt.show()

	plt.plot(x, greedyD2load, label='uniform D=2 multi-choice')
	plt.plot(x, gridD2load, label='uniform D=2 grid local search')
	plt.plot(x, adaptiveload, label='uniform adaptive')
	plt.title('Comparing Greedy, Grid, Adaptive for Maximum Load')
	plt.legend(loc='upper left')
	plt.xlabel('Trials')
	plt.ylabel('Max Load')
	plt.show()

	plt.plot(x, greedyD2a, label='uniform D=2 multi-choice')
	plt.plot(x, gridD2a, label='uniform D=2 grid local search')
	plt.plot(x, adaptivea, label='uniform adaptive')
	plt.title('Comparing Greedy, Grid, Adaptive for Allocation Time')
	plt.legend(loc='upper left')
	plt.xlabel('Trials')
	plt.ylabel('Allocation Time')
	plt.show()

def main():
	# set up standard experiment parameters
	trials = 10000
	n = 100
	f = 1
	results = []
	settings = Settings(balls=n*f, bins=n, capacity=n*f)

	print "f = ", f
	for t in range(trials):
		if t % 100 == 0:
			print "trial", t

		experiments = [] 
		experiments.append(Experiment("uniform D=1 greedy", settings, Greedy(d=1)))
		experiments.append(Experiment("uniform D=2 greedy", settings, Greedy(d=2)))
		experiments.append(Experiment("uniform D=2 left", settings, Left(d=2)))
		experiments.append(Experiment("uniform D=1 grid", settings, Grid(d=1)))
		experiments.append(Experiment("uniform D=2 grid", settings, Grid(d=2)))
		experiments.append(Experiment("uniform D=3 grid", settings, Grid(d=3)))
		experiments.append(Experiment("uniform adaptive", settings, Adaptive()))
		# limited capacity (with weight penalty); with local search; with graph structure; with non-uniformity.

		# run trials on each experiment
		for experiment in experiments:
			experiment.run()
			results.append(experiment.getResults())

	# display results
	print "display results"
	displayResults(results, trials)


main()
