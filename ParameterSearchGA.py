
from Hunter import PopulationInitialization
import numpy as np
import matplotlib.pyplot as plt
import time
#from PopulationClass import PopulationInitialization

class GeneticAlgorithm():

    def __init__(self, populationSize = 10, numberOfGenes = 11, crossoverProb = 0.8, mutationProb = 0.01, tourProb = 0.75, variableRange = 3, numberOfGenerations = 100):
        self.populationSize = populationSize
        self.numberOfGenes = numberOfGenes
        self.population = np.zeros([self.populationSize, self.numberOfGenes])
        self.crossoverProbability = crossoverProb
        self.mutationProbability = mutationProb

        self.pTournament = tourProb
        self.variableRange = variableRange
        self.numberOfGenerations = numberOfGenerations
        self.fitness = np.zeros(self.populationSize)

        #Constraints parameters
        self.forestLower = 40
        self.forestUpper = 50
        self.wolfLower = 15
        self.wolfUpper = 45
        self.mooseLower = 20
        self.mooseUpper = 90
        self.vegDensityLower = 0.001
        self.vegDensityUpper = 0.5
        self.hungerDeathLower = 100
        self.hungerDeathUpper = 100
        self.wolfHungerRedLower = 100
        self.wolfHungerRedUpper = 100
        self.mooseHungerRedLower = 10
        self.mooseHungerRedUpper = 10
        self.mooseReproLower = 10
        self.mooseReproUpper = 25
        self.wolfReproLower = 4
        self.wolfReproUpper = 9
        self.mooseVisionMateLower = 1
        self.mooseVisionMateUpper = 12
        self.mooseVisionDangerLower = 1
        self.mooseVisionDangerUpper = 6
        self.wolfVisionMateLower = 1
        self.wolfVisionMateUpper = 6
        self.wolfVisionFoodLower = 1
        self.wolfVisionFoodUpper = 8
        self.numberOfHuntersLower = 1
        self.numberOfHuntersUpper = 50
        self.huntingLengthLower = 1
        self.huntingLengthUpper =25
        self.huntCountLower = 1
        self.huntCountUpper = 500


    #Number of genes will be the same as number of paramters in rea number encoding
    #Initialize to uniform random number
    def InitializePopulation(self):
            self.population = np.random.random(self.population.shape)
    #Everything is treated as floats until sent to simulation
    #Here we can introduce dependence on previous assignment, eg. forest = 40, then
    # number of wolfs/moose is dependent on athat value
    def DecodeChromosome(self,chromosome):
        x = np.zeros(self.numberOfGenes)
        for i in range(len(x)):
            #Encode forest size
            if i == 0:
                x[i] = (self.forestLower+(self.forestUpper-self.forestLower)*chromosome[i])
            #Inital number of mooses
            elif i == 1:
                x[i] = (self.mooseLower+((self.mooseUpper-self.mooseLower)*chromosome[i]))
            elif i == 2:
                x[i] = (self.wolfLower+((self.wolfUpper-self.wolfLower)*chromosome[i]))
            #Vegetation density
            elif i == 3:
                x[i] = self.vegDensityLower+(self.vegDensityUpper-self.vegDensityLower)*chromosome[i]
            #Hunger death
            elif i == 4:
                x[i] = (self.hungerDeathLower+(self.hungerDeathUpper-self.hungerDeathLower)*chromosome[i])
            #Wolf hunger reduction
            elif i == 5:

                x[i] = (self.wolfHungerRedLower+(self.wolfHungerRedUpper-self.wolfHungerRedLower)*chromosome[i])
            #Moose hunger reduction
            elif i == 6:
                x[i] = (self.mooseHungerRedLower+(self.mooseHungerRedUpper-self.mooseHungerRedLower)*chromosome[i])
            #mooseOffspringProbability
            elif i == 7:
                x[i] = chromosome[i]
            #wolfOffspringProbability
            elif i == 8:
                x[i] = chromosome[i]
            #mooseVisionMate
            elif i == 9:
                x[i] = (self.mooseVisionMateLower+(self.mooseVisionMateUpper-self.mooseVisionMateLower)*chromosome[i])
            #mooseVisionDanger
            elif i == 10:
                x[i] = (self.mooseVisionDangerLower+(self.mooseVisionDangerUpper-self.mooseVisionDangerLower)*chromosome[i])
                #wolfVisionMate
            elif i == 11:
                x[i] = (self.wolfVisionMateLower+(self.wolfVisionMateUpper-self.wolfVisionMateLower)*chromosome[i])
            #wolfVisonDanger
            elif i == 12:
                x[i] = (self.wolfVisionFoodLower+(self.wolfVisionFoodUpper-self.wolfVisionFoodLower)*chromosome[i])
            elif i == 13 : 
                x[i] = (self.numberOfHuntersLower+(self.numberOfHuntersUpper-self.numberOfHuntersLower)*chromosome[i])
            elif i == 14 :
                x[i] = (self.huntingLengthLower+(self.huntingLengthUpper-self.huntingLengthLower)*chromosome[i])
            elif i == 15:
                x[i] = (self.huntCountLower+(self.huntCountUpper-self.huntCountLower)*chromosome[i])

        return x

    def EvaluateIndividual(self,x):
        forrestSize = int(x[0])
        numberOfMooses = int(x[1])
        numberOfWolfs = int(x[2])
        vegetationDensity = x[3]
        hungerDeath = int(x[4])
        wolfHungerReduction = int(x[5])
        mooseHungerReduction = int(x[6])
        #mooseReproAge = int(x[7])
        #wolfReproAge = int(x[8])
        mooseOffspringProbability = x[7]
        wolfOffspringProbability = x[8]
        mooseVisionMate = int(x[9])
        mooseVisionDanger = int(x[10])
        wolfVisionMate = int(x[11])
        wolfVisionFood = int(x[12])
        numberOfHunters = int(x[13])
        huntingLength = int(x[14])
        huntCount = int(x[15])


        totalTime = 900
        plottingOn = False
        ecosystem = PopulationInitialization(forrestSize,
                                             numberOfMooses, numberOfWolfs,
                                             vegetationDensity, hungerDeath,
                                             wolfHungerReduction,mooseHungerReduction,
                                             mooseOffspringProbability, wolfOffspringProbability,
                                            mooseVisionMate , mooseVisionDanger,
                                            wolfVisionMate, wolfVisionFood,
                                             numberOfHunters, huntingLength,huntCount,
                                             totalTime, plottingOn)

        ecosystem.InitializeMooses()
        ecosystem.InitializeWolfs()
        #Fitness value is the time run, simulation breaks when pop is zero or over populated
        f=ecosystem.RunSimulation()
        return f

    def TournamentSelect(self):
        iTmp1 = int(np.floor(np.random.random()*self.populationSize))
        iTmp2 = int(np.floor(np.random.random()*self.populationSize))

        r = np.random.random()
        if r < self.pTournament:
            if self.fitness[iTmp1] > self.fitness[iTmp2]:
                iSelected = iTmp1
            else:
                iSelected = iTmp2
        else :
            if self.fitness[iTmp1] > self.fitness[iTmp2]:
                iSelected = iTmp2
            else :
                iSelected = iTmp1
        return iSelected

    def Cross(self,chromosome1, chromosome2):
        crossoverPoint = int(np.floor(np.random.random()*(self.numberOfGenes-1)))
        newChromosomePair = np.zeros([2,self.numberOfGenes])
        for j in range(self.numberOfGenes):
            if j <= crossoverPoint:
                newChromosomePair[0,j] = chromosome1[j]
                newChromosomePair[1,j] = chromosome2[j]
            else:
                newChromosomePair[0,j] = chromosome2[j]
                newChromosomePair[1,j] = chromosome1[j]
        return newChromosomePair
    #Mutation could be changed to creep mutation
    def Mutate(self, chromosome):
        mutatedChomosome = chromosome
        for j in range(self.numberOfGenes):
            r = np.random.random()
            if r < self.mutationProbability:
                mutatedChomosome[j] = np.random.random()
        return mutatedChomosome

    #Main loop for GA
    def RunOptimization(self):

        maximumFitness = 0
        fitness = np.zeros(self.fitness.shape)
        xBest = np.zeros([1,self.numberOfGenes])
        bestIndividualIndex = 0
        decodedPopulation = np.zeros([self.populationSize,self.numberOfGenes])
        for iGeneration in range(self.numberOfGenerations):
            for i in range(self.populationSize):
                chromosome = self.population[i,:]
                x = self.DecodeChromosome(chromosome)
                f = self.EvaluateIndividual(x)
                fitness[i] = f
                decodedPopulation[i,:] = x.copy()

                if fitness[i] > maximumFitness:
                    maximumFitness = fitness[i]
                    bestIndividualIndex = i
                    xBest = x.copy()
            self.fitness = fitness.copy()
            tempPopulation = self.population.copy()

            for i in range(0,self.populationSize,2):
                i1 = self.TournamentSelect()
                i2 = self.TournamentSelect()
                chromosome1 = self.population[i1,:].copy()
                chromosome2 = self.population[i2,:].copy()

                r = np.random.random()
                if r < self.crossoverProbability:
                    newChromosomePair = self.Cross(chromosome1, chromosome2)
                    tempPopulation[i,:] = newChromosomePair[0,:].copy()
                    tempPopulation[i+1,:] = newChromosomePair[1,:].copy()
                else:
                    tempPopulation[i,:] = chromosome1.copy()
                    tempPopulation[i+1,:] = chromosome2.copy()


            for i in range(self.populationSize):
                originalChromosome = tempPopulation[i,:].copy()
                mutatedChromosome = self.Mutate(originalChromosome)
                tempPopulation[i,:] = mutatedChromosome.copy()

            tempPopulation[0,:] = self.population[bestIndividualIndex,:]
            self.population = tempPopulation.copy()
            print(f'Generation: {iGeneration}')
            print(f'xBest = {xBest}')
            print(f'maximumFitness = {maximumFitness}')
        print(f'xBest = {xBest}')
        print(f'maximumFitness = {maximumFitness}')

if __name__ == "__main__":

    ga = GeneticAlgorithm(populationSize = 30, numberOfGenes = 16, crossoverProb=0.75, mutationProb = 0.12, tourProb = 0.80, variableRange = 3, numberOfGenerations = 1000)
    ga.InitializePopulation()
    ga.RunOptimization()