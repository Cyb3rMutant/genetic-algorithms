# purpose:  genetic algorithm to optimize functions with selection, crossover and mutation operators
# module: artificial intelligence ii
# author: Yazeed Abu-Hummos
# UWE ID: 21014295
# last edit time and date: 03:24 05/12/2022


# note: uncomment either of the functions inside of the Individual class and change the min and max gene values


from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy as np


NUM_GENES = 20
POPUL_SIZE = 30
NUM_GENERATIONS = 150
MUTATION_RATE = 0.1
MUTATION_STEP_RANGE = 0.8
MIN_GENE = -100
MAX_GENE = 100
ROWS = 2
COLS = 2
best_fitness = np.empty(NUM_GENERATIONS, dtype=float)
avg_fitness = np.empty(NUM_GENERATIONS, dtype=float)


class Individual():
    def __init__(self):
        self.chromosome = np.zeros(NUM_GENES, dtype=float)
        self.fitness = np.inf

    def calc_fitness(self):
        # self.fitness = 0
        # sum_1 = 0
        # sum_2 = 0
        # for i in range(NUM_GENES):
        #     sum_1 += self[i]**2
        #     sum_2 += 0.5*i * self[i]

        # self.fitness += sum_1+(sum_2**2)+sum_2**4

        self.fitness = 0
        for i in range(NUM_GENES-1):
            self.fitness += (100*(self[i+1]-self[i]**2)**2)+(
                (1-self[i])**2)

    def __getitem__(self, idx):
        return self.chromosome[idx]

    def __setitem__(self, idx, val):
        self.chromosome[idx] = val


class Population():

    def __init__(self):
        self.list = np.array([Individual()
                             for i in range(POPUL_SIZE)], dtype=Individual)
        self.best = Individual()
        self.__gen = -1

    def __getitem__(self, idx):
        return self.list[idx]

    def __setitem__(self, idx, val):
        self.list[idx] = val

    def __calc_mutation_step(self):
        x = self.__gen
        p = (MAX_GENE-MIN_GENE)/2
        return p/((
            ((0.01*(x**(math.e))) / (math.e * math.sqrt(p)))) + (math.e)) * (-1 if np.random.randint(0, 2) else 1)

    def init(self):
        self.__gen = -1

        for ind in self.list:
            ind.chromosome = np.random.uniform(
                MIN_GENE, MAX_GENE, NUM_GENES)
            ind.calc_fitness()
            if (self.best.fitness > ind.fitness):
                self.best = deepcopy(ind)

    def __tournament_selection(self):
        offspring = Population()

        for selected in range(POPUL_SIZE):

            index_1 = np.random.randint(0, POPUL_SIZE)
            index_2 = np.random.randint(0, POPUL_SIZE)
            if (self[index_1].fitness < self[index_2].fitness):
                offspring[selected] = deepcopy(self[index_1])
            else:
                offspring[selected] = deepcopy(self[index_2])
        self.__gen += 1
        self.list = np.copy(offspring.list)

    def __roulette_wheel_selection(self):
        total_fitness = 0.0
        for ind in self.list:
            total_fitness += ind.fitness

        ind_probability = np.array(
            [ind.fitness/total_fitness for ind in self.list])
        ind_probability = 1/ind_probability
        inverse_sum = sum(ind_probability)
        ind_probability = ind_probability / inverse_sum

        offspring = Population()

        for selected in range(POPUL_SIZE):
            offspring[selected] = deepcopy(
                np.random.choice(self.list, p=ind_probability))

        self.__gen += 1
        self.list = np.copy(offspring.list)

    def selection(self, type="tournament"):
        func = {"tournament": self.__tournament_selection,
                "roulette_wheel": self.__roulette_wheel_selection}
        func[type]()

    def __one_point_crossover(self):
        for i in range(0, POPUL_SIZE, 2):
            crossover_point = np.random.randint(NUM_GENES//8, NUM_GENES//2)

            chromosome_1 = np.copy(self[i][-crossover_point:])
            chromosome_2 = np.copy(self[i+1][-crossover_point:])

            self[i][-crossover_point:] = np.copy(chromosome_2)
            self[i+1][-crossover_point:] = np.copy(chromosome_1)

    def __uniform_crossover(self):
        for i in range(0, POPUL_SIZE, 2):
            chromosome_1 = np.copy(self[i].chromosome)
            chromosome_2 = np.copy(self[i+1].chromosome)
            for j in range(NUM_GENES):
                if (np.random.uniform() >= 0.5):
                    self[i][j] = np.copy(chromosome_2[j])
                    self[i+1][j] = np.copy(chromosome_1[j])

    def __arithmatic_crossover(self):
        for i in range(0, POPUL_SIZE, 2):
            alpha = np.random.uniform()
            chromosome_1 = np.copy(self[i].chromosome)
            chromosome_2 = np.copy(self[i+1].chromosome)
            for j in range(NUM_GENES):
                self[i][j] = (alpha*chromosome_1[j]) + \
                    ((1-alpha)*chromosome_2[j])
                self[i+1][j] = (alpha*chromosome_2[j]) + \
                    ((1-alpha)*chromosome_1[j])

    def crossover(self, type="one_point"):
        func = {"one_point": self.__one_point_crossover,
                "uniform": self.__uniform_crossover,
                "arithmatic": self.__arithmatic_crossover}
        func[type]()

    def __static_mutation(self):
        for ind in self.list:
            for gene in range(NUM_GENES):
                if (np.random.uniform() < MUTATION_RATE):
                    ind[gene] += np.random.uniform(MUTATION_STEP_RANGE) * \
                        (-1 if np.random.randint(0, 2) else 1)

                    if (ind[gene] > MAX_GENE):
                        ind[gene] = MAX_GENE-0.00001
                    if (ind[gene] < MIN_GENE):
                        ind[gene] = MIN_GENE

    def __degrading_mutation(self):
        mutation_step = self.__calc_mutation_step()
        for ind in self.list:
            for gene in range(NUM_GENES):
                if (np.random.uniform() < MUTATION_RATE):
                    ind[gene] += mutation_step

                    if (ind[gene] > MAX_GENE):
                        ind[gene] = MAX_GENE-0.00001
                    if (ind[gene] < MIN_GENE):
                        ind[gene] = MIN_GENE

    def mutation(self, type="static"):
        func = {"static": self.__static_mutation,
                "degrading": self.__degrading_mutation}
        func[type]()

    def stats(self):
        current_best_idx = 0
        current_worst_idx = 0

        for i in range(POPUL_SIZE):
            self[i].calc_fitness()

            # Calc total fitness
            avg_fitness[self.__gen] += self[i].fitness

            # find current best
            if (self[current_best_idx].fitness > self[i].fitness):
                current_best_idx = i

            # find current worst
            if (self[current_worst_idx].fitness < self[i].fitness):
                current_worst_idx = i

        # deviding total fitness by population size to get avg
        avg_fitness[self.__gen] /= POPUL_SIZE

        self.list[current_worst_idx] = deepcopy(self.best)

        # replacing current worst with previous best to not lose it in case it's the best best
        if (self.best.fitness > self[current_best_idx].fitness):
            self.best = deepcopy(self[current_best_idx])

        best_fitness[self.__gen] = self.best.fitness


fig, axs = plt.subplots(ROWS, COLS, figsize=(16, 9))

for i in range(ROWS):
    for j in range(COLS):
        best_fitness = np.zeros(NUM_GENERATIONS, dtype=float)
        avg_fitness = np.zeros(NUM_GENERATIONS, dtype=float)
        population = Population()
        population.init()
        for gen in range(NUM_GENERATIONS):
            population.selection()
            population.crossover()
            population.mutation("degrading")

            population.stats()

        print(population.best.chromosome)
        print(best_fitness[-1])

        axs[i, j].set_xlabel("best fitness= %f" % best_fitness[-1])
        axs[i, j].plot(avg_fitness, label="avg")
        axs[i, j].plot(best_fitness, label="best")
        axs[i, j].legend()

plt.suptitle(f"NUM_GENES = {NUM_GENES} || POPUL_SIZE = {POPUL_SIZE} || NUM_GENERATIONS = {NUM_GENERATIONS}\nMUTATION_RATE = {MUTATION_RATE} || MIN_GENE = {MIN_GENE} || MAX_GENE = {MAX_GENE}\ntest function = ROSENBROCK FUNCTION")
plt.show()
