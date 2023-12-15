#Import packages
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
# from numba.experimental import jitclass
import time
from IPython import display

class PopulationInitialization():
    #Set parameters
    def __init__(self,forrestSize=10, numberOfMooses=0, numberOfWolfs=0,
                 vegetationDenstity=0.1, hungerDeath=100,
                 wolfHungerReduction=100, mooseHungerReduction=100,
                 mooseOffspringProbability=0, wolfOffspringProbability=0,
                 mooseVisionMate = 2, mooseVisionDanger = 2,
                 wolfVisionMate =2, wolfVisionFood =2, numberOfHunters=5,
                 huntingLength = 5, huntCount = 5,
                totalTime=0, plottingOn=False):
        #Initial condition of populations and forrest size
        self.forrestSize = forrestSize
        self.initialNumberOfMooses = numberOfMooses
        self.initialNumberOfWolfs = numberOfWolfs
        self.numberOfMooses = numberOfMooses
        self.numberOfWolfs = numberOfMooses
        self.wolfDeathAge = 13         #To be defined better, Wolves can live up to 13 years in the wild
        self.mooseDeathAge = 25        #To be defined better, Moose may live up to 25 years
        self.vegetationDenstity = vegetationDenstity    #To be defined better
        self.hungerDeath = hungerDeath           #To be defined better
        self.wolfHungerReduction = wolfHungerReduction    #To be defined better
        self.mooseHungerReduction = mooseHungerReduction   #To be defined better
        self.mooseReproAge = 11 #To be defined better
        self.wolfReproAge = 9 #To be defined better
        self.initialNumberOfHunters = numberOfHunters
        self.numberOfHunters = numberOfHunters #To be defined better
        self.hunting = True
        self.huntingLength = huntingLength
        self.huntCount = huntCount

        #Parameters controling dynamics of model
        self.initialMooseOffspringProbability = mooseOffspringProbability
        self.initialWolfOffspringProbability = wolfOffspringProbability
        self.mooseOffspringProbability = mooseOffspringProbability
        self.wolfOffspringProbability = wolfOffspringProbability
        self.mooseVisionMate = mooseVisionMate
        self.mooseVisionDanger = mooseVisionDanger
        self.wolfVisionMate = wolfVisionMate
        self.wolfVisionFood = wolfVisionFood
        self.huntedSpecies = 1

        #Setup vectors and arrays for model
        self.totalTime = int(totalTime)
        self.timeVector = np.arange(0,self.totalTime)
        self.populationVector = np.zeros([3,self.totalTime]) #Hunter Update
        self.forrest = np.zeros([self.forrestSize,self.forrestSize])
        self.animalAge = np.zeros([self.forrestSize,self.forrestSize]) #Added age matrix
        self.vegetation = np.zeros([self.forrestSize,self.forrestSize]) #Added vegetation matrix
        self.hunger = np.zeros([self.forrestSize,self.forrestSize]) #Added hunger matrix
        #Sets if simulation should plot each time step during run time
        self.plottingOn = plottingOn

    #Define functions
    #Give prey random initial positions in forrest
    def InitializeMooses(self):
        counter = 0
        while counter < self.initialNumberOfMooses:
            i = np.random.randint(0,self.forrest.shape[0])
            j = np.random.randint(0,self.forrest.shape[1])
            if self.forrest[i,j] == 0:
                self.forrest[i,j] = 1
                counter += 1

    #Give predators random initial positions in forrest
    def InitializeWolfs(self):
        counter = 0
        while counter < self.initialNumberOfWolfs:
            i = np.random.randint(0,self.forrest.shape[0])
            j = np.random.randint(0,self.forrest.shape[1])
            if self.forrest[i,j] == 0:
                self.forrest[i,j] = 2
                counter += 1

    #Init Hunters
    def InitializeHunters(self):
        counter = 0
        while counter < self.numberOfHunters:
            i = np.random.randint(0,self.forrest.shape[0])
            j = np.random.randint(0,self.forrest.shape[1])
            if self.forrest[i,j] == 0:
                self.forrest[i,j] = 3
                counter += 1

    #Update hunger and age matrices
    def UpdateHungerAndAge(self, new, old):
            assert new != old, 'Update error in hunger/age, new position cannot be same'
            self.hunger[new] = self.hunger[old]
            self.hunger[old] = 0
            self.animalAge[new] = self.animalAge[old]
            self.animalAge[old] = 0
#Moving strategies
    def FindMate(self,visionMate,i,j,n, species):
            if np.any(self.forrest[(i+2)%n:(i+visionMate)%n,j]) == species and self.forrest[(i+1)%n,j] == 0 :
                self.forrest[(i+1)%n,j] = species
                self.UpdateHungerAndAge(((i+1)%n,j) , (i,j))
                return 1
            elif np.any(self.forrest[i-2,j:(i-visionMate)] == species) and self.forrest[i-1,j] == 0:
                self.forrest[i-1,j] = species
                self.UpdateHungerAndAge((i-1,j) , (i,j))
                return 1
            elif np.any(self.forrest[i,(j+2)%n:(j-visionMate)] == species) and self.forrest[i,(j+1)%n] == 0:
                self.forrest[i,(j+1)%n] = species
                self.UpdateHungerAndAge((i,(j+1)%n) , (i,j))
                return 1
            elif np.any(self.forrest[i,j-2:(j-visionMate)] == species) and self.forrest[i,j-1]==0:
                self.forrest[i,j-1] = species
                self.UpdateHungerAndAge((i,j-1) , (i,j))
                return 1
            return 0
    def RunFromPredator(self,visionDanger,i,j,n):
            if np.any(self.forrest[(i+2)%n:(i+visionDanger)%n,j]) == 2 and self.forrest[(i+1)%n,j] == 0 :
                self.forrest[(i-1),j] = 1
                self.UpdateHungerAndAge(((i-1),j) , (i,j))
                return 1
            elif np.any(self.forrest[i-2,j:(i-visionDanger)] == 2) and self.forrest[i-1,j] == 0:
                self.forrest[(i+1)%n,j] = 1
                self.UpdateHungerAndAge(((i+1)%n,j) , (i,j))
                return 1
            elif np.any(self.forrest[i,(j+2)%n:(j-visionDanger)] == 2) and self.forrest[i,(j+1)%n] == 0:
                self.forrest[i,(j-1)] = 1
                self.UpdateHungerAndAge((i,(j-1)) , (i,j))
                return 1
            elif np.any(self.forrest[i,j-2:(j-visionDanger)] == 2) and self.forrest[i,j-1]==0:
                self.forrest[i,(j+1)%n] = 1
                self.UpdateHungerAndAge((i,(j+1)%n) , (i,j))
                return 1
            return 0
    def WalkRandomly(self,i,j,n,species):
            direction = np.random.random()
            if direction < 0.25 and self.forrest[i,j-1] == 0:
                self.forrest[i,j-1] = species
                self.UpdateHungerAndAge((i,j-1) , (i,j))
                return 1
            #Walk east
            elif direction < 0.50 and self.forrest[i,(j+1)%n] == 0:
                self.forrest[i,(j+1)%n] = species
                self.UpdateHungerAndAge((i,(j+1)%n) , (i,j))
                return 1
            #Walk north
            elif direction < 0.75 and self.forrest[(i+1)%n,j] == 0 :
                self.forrest[(i+1)%n,j] = species
                self.UpdateHungerAndAge(((i+1)%n,j) , (i,j))
                return 1
            elif direction < 1 and self.forrest[i-1,j] == 0:
                self.forrest[i-1,j] = species
                self.UpdateHungerAndAge((i-1,j) , (i,j))
                return 1
            #Stay where you are!
            else:
                self.forrest[i,j] = species
                return 1
            return 0
    def ChasePrey(self,visionFood, i,j,n):
            if np.any(self.forrest[(i+2)%n:(i+visionFood)%n,j]) == 1 and self.forrest[(i+1)%n,j] == 0 :
                self.forrest[(i+1)%n,j] = 2
                self.UpdateHungerAndAge(((i+1)%n,j) , (i,j))
                return 1
            elif np.any(self.forrest[i-2:(i-visionFood)] == 1) and self.forrest[i-1,j] == 0:
                self.forrest[i-1,j] = 2
                self.UpdateHungerAndAge((i-1,j) , (i,j))
                return 1
            elif np.any(self.forrest[i,(j+2)%n:(j-visionFood)] == 1) and self.forrest[i,(j+1)%n] == 0:
                self.forrest[i,(j+1)%n] = 2
                self.UpdateHungerAndAge((i,(j+1)%n) , (i,j))
                return 1
            elif np.any(self.forrest[i,j-2:(j-visionFood)] == 1) and self.forrest[i,j-1]==0:
                self.forrest[i,j-1] = 2
                self.UpdateHungerAndAge((i,j-1) , (i,j))
                return 1
            return 0

    #Move the prey in forrest
    def MoveMooses(self):
        species = 1
        n = self.forrest.shape[0]
        moosePosition = zip(*np.where(self.forrest == 1))
        #Loop over all position, check enviroment and make a move for each moose
        for i,j in moosePosition:
            #Leave current position
            self.forrest[i,j] = 0
            visionMate = self.mooseVisionMate
            visionDanger = self.mooseVisionDanger
            strategies = np.arange(0,3,1)
            #strategies = np.ones(3)*2
            #np.random.shuffle(strategies)
            #Loop over strategies, if step taken retun code 1 (True) breaks loop
            for s in strategies:
            #First find a mate, and take one step toward it
                if s == 0:
                    if self.FindMate(visionMate,i,j,n,species):
                        break
            #If no mate run from prey
                elif s == 1:
                    if self.RunFromPredator(visionDanger,i,j,n):
                        break
            #No predator in sight walk randomly!
                elif s==2:
                    if self.WalkRandomly(i,j,n,species):
                        break
    #Prey reproduces
    def MoosesGiveBirth(self):
        n = self.forrest.shape[0]
        moosePosition = zip(*np.where(self.forrest == 1))
        for i,j in moosePosition:
            #If there is a mate nearby
            thereIsAMate = (self.forrest[i-1,j] == 1  or self.forrest[(i+1)%n,j] == 1 or
                self.forrest[i,(j+1)%n] == 1 or self.forrest[i,j-1] == 1);
            thereIsOffspring = np.random.random() < self.mooseOffspringProbability
            thereIsSuitableAge = self.animalAge[i,j] >= self.mooseReproAge
            thereIsAMate=1
            if thereIsAMate and thereIsOffspring and thereIsSuitableAge:
                self.SpawnNewbornMoose()
    #Place a new prey andomly in the forrest
    def SpawnNewbornMoose(self):
        #Get non occupied slots
        birthplaces =  np.where(self.forrest == 0)
        #The grid can become overpopulated, if so do nothing
        if birthplaces[0].size != 0:
                # Set condition to select a slot randomly with equal prob
                fractionOfEmptySlots =  1/np.shape(birthplaces[0])[0]
                condition = fractionOfEmptySlots
                r = np.random.random()
                for i,j in zip(*birthplaces):
                        if r < condition:
                            self.forrest[i,j] = 1
                            break
                        condition += fractionOfEmptySlots

    #Move the predator in the forrest
    def MoveWolfs(self):
        species =2
        n = self.forrest.shape[0]
        wolfPosition = zip(*np.where(self.forrest == 2))
        #Loop over all position, check enviroment and make a move for each moose
        for i,j in wolfPosition:
            #Leave current position
            self.forrest[i,j] = 0
            species = 2
            #Take a random step
            visionFood = self.wolfVisionFood
            visionMate = self.wolfVisionMate
            strategies = np.arange(0,3,1)
            #np.random.shuffle(strategies)
            #Make it use only random
            #strategies = np.ones(3)*2
            #Loop over strategies, if step taken retun code 1 (True) breaks loop
            for s in strategies:
            #First find a mate, and take one step toward it
                if s == 0:
                    if self.FindMate(visionMate,i,j,n,species):
                        break
            #If no mate run from prey
                elif s == 1:
                    if self.ChasePrey(visionFood,i,j,n):
                        break
            #No predator in sight walk randomly!
                elif s == 2:
                    if self.WalkRandomly(i,j,n,species):
                        break


    def MooseEat(self):
        '''
            If the moose stands on a tile with vegetation then it may consume
        '''
        n = self.forrest.shape[0]
        moosePosition = zip(*np.where(self.forrest == 1))
        for i,j in moosePosition:
            if self.vegetation[i,j] >= 0:
                self.vegetation[i,j] -= 2
                self.hunger[i,j] -= self.mooseHungerReduction

    #Predator eats prey if it movesd to position adjacent to prey
    def WolfsEat(self):
        n = self.forrest.shape[0]
        wolfPosition = zip(*np.where(self.forrest == 2))
        for i,j in wolfPosition:
            #If there is food, eat it!
            if self.forrest[i-1,j] == 1 :
                self.forrest[i-1,j] = 0
                self.hunger[i,j] -= self.wolfHungerReduction
            elif  self.forrest[(i+1)%n,j] == 1:
                self.forrest[(i+1)%n,j] = 0
                self.hunger[i,j] -= self.wolfHungerReduction
            elif  self.forrest[i,(j+1)%n] == 1:
                self.forrest[i,(j+1)%n] = 0
                self.hunger[i,j] -= self.wolfHungerReduction
            elif  self.forrest[i,j-1] == 1:
                self.forrest[i,j-1] = 0
                self.hunger[i,j] -= self.wolfHungerReduction

    #Predator reproduces
    def WolfsGiveBirth(self):
        n = self.forrest.shape[0]
        wolfPosition = zip(*np.where(self.forrest == 2))
        for i,j in wolfPosition:
            #If there is a mate nearby
            thereIsAMate = (self.forrest[i-1,j] == 2  or self.forrest[(i+1)%n,j] == 2 or
                self.forrest[i,(j+1)%n] == 2 or self.forrest[i,j-1] == 2);
            thereIsOffspring = np.random.random() < self.wolfOffspringProbability
            thereIsSuitableAge = self.animalAge[i,j] >= self.wolfReproAge
            thereIsAMate=1
            if thereIsAMate and thereIsOffspring and thereIsSuitableAge:
                self.SpawnNewbornWolf()

    #Assign a random position to new predator
    def SpawnNewbornWolf(self):
        #Get non occupied slots
        birthplaces =  np.where(self.forrest == 0)
        #The grid can become overpopulated
        if birthplaces[0].size != 0:
                # Set condition to select a slot randomly with equal prob
                fractionOfEmptySlots =  1/np.shape(birthplaces[0])[0]
                condition = fractionOfEmptySlots
                r = np.random.random()
                for i,j in zip(*birthplaces):
                        if r < condition:
                            self.forrest[i,j] = 2
                            self.hunger[i,j] = 50
                            break
                        condition += fractionOfEmptySlots

    #Function to update age and natural death of both prey and predator
    def AnimalDies(self):
        # n = self.forrest.shape[0]
        wolfPosition = zip(*np.where(self.forrest == 2))
        moosePosition = zip(*np.where(self.forrest == 1))
        for i,j in wolfPosition:
            self.animalAge[i,j] += 1
            if self.animalAge[i,j] >= self.wolfDeathAge or self.hunger[i,j] >= self.hungerDeath:
                assert 1 == 1, 'this should never be called with these parameters'
                self.forrest[i,j] = 0
                self.animalAge[i,j] = 0
                self.hunger[i,j] = 0
        for i,j in moosePosition:
            self.animalAge[i,j] += 1
            if self.animalAge[i,j] >= self.mooseDeathAge or self.hunger[i,j] >= self.hungerDeath:
                assert 1 == 1, 'this should never be called with these parameters'
                self.forrest[i,j] = 0
                self.animalAge[i,j] = 0
                self.hunger[i,j] = 0

    def SpawnVegetation(self):
        numberOfVegetation = round(self.vegetationDenstity*self.forrestSize*self.forrestSize)
        for count in range(numberOfVegetation):
            i = np.random.randint(0,self.forrest.shape[0])
            j = np.random.randint(0,self.forrest.shape[1])
            self.vegetation[i,j] += 1

    def IncreaseHunger(self):
        wolfPosition = zip(*np.where(self.forrest == 2))
        moosePosition = zip(*np.where(self.forrest == 1))
        for i,j in wolfPosition:
            self.hunger[i,j] += 10
        for i,j in moosePosition:
            self.hunger[i,j] += 10

    def MoveHunters(self):
        n = self.forrest.shape[0]
        hunterPosition = zip(*np.where(self.forrest == 3))
        #Loop over all position, check enviroment and make a move for each hunter
        for i,j in hunterPosition:
            #Leave current position
            self.forrest[i,j] = 0
            #Take a random step
            direction = np.random.random()
            visionFood = 4
            #Chase a prey
            if np.any(self.forrest[(i+2)%n:(i+visionFood)%n,j]) == 2 and self.forrest[(i+1)%n,j] == 0 :
                self.forrest[(i+1)%n,j] = 3
            elif np.any(self.forrest[i-2,j:(i-visionFood)] == 2) and self.forrest[i-1,j] == 0:
                self.forrest[i-1,j] = 3
            elif np.any(self.forrest[i,(j+2)%n:(j-visionFood)] == 2) and self.forrest[i,(j+1)%n] == 0:
                self.forrest[i,(j+1)%n] = 3
            elif np.any(self.forrest[i,j-2:(j-visionFood)] == 2) and self.forrest[i,j-1]==0:
                self.forrest[i,j-1] = 3
            #No prey in sight walk randomly!
            #Walk west
            elif direction < 0.25 and self.forrest[i,j-1] == 0:
                self.forrest[i,j-1] = 3
            #Walk east
            elif direction < 0.50 and self.forrest[i,(j+1)%n] == 0:
                self.forrest[i,(j+1)%n] = 3
            #Walk north
            elif direction < 0.75 and self.forrest[(i+1)%n,j] == 0 :
                self.forrest[(i+1)%n,j] = 3
            elif direction < 1 and self.forrest[i-1,j] == 0:
                self.forrest[i-1,j] = 3
            #Stay where you are!
            else:
                self.forrest[i,j] = 3

    def Hunting(self):
        n = self.forrest.shape[0]
        huntCount = 0
        hunterPosition = zip(*np.where(self.forrest == 3))
        for i,j in hunterPosition:
            if huntCount <= self.huntCount:
                if self.forrest[i-1,j] ==  self.huntedSpecies :
                    self.forrest[i-1,j] = 0
                    huntCount += 1
                elif  self.forrest[(i+1)%n,j] ==  self.huntedSpecies :
                    self.forrest[(i+1)%n,j] = 0
                    huntCount += 1
                elif  self.forrest[i,(j+1)%n] ==  self.huntedSpecies :
                    self.forrest[i,(j+1)%n] = 0
                    huntCount += 1
                elif  self.forrest[i,j-1] ==  self.huntedSpecies :
                    self.forrest[i,j-1] = 0
                    huntCount += 1
            else:
                huntCount = 0
                continue

    #Remove hunters after hunting done
    def RemoveHunters(self):
        hunterPosition = zip(*np.where(self.forrest == 3))
        #Find hunters
        for i,j in hunterPosition:
            #Remove Hunters
            self.forrest[i,j] = 0

    #Count total population at each time
    def UpdatePopulation(self):
        self.numberOfMooses = np.sum(self.forrest == 1)
        self.numberOfWolfs = np.sum(self.forrest == 2)

    # Run simulation over all time steps
    def RunSimulation(self):
        if self.plottingOn == True:
            plotter = Plotting()
        counter = 0 #Hunter Update
        f = 0
        for tIteration in range(self.totalTime):
            #Events
            '''
                Might want a second opinion on this order, i just put hunger and mooseeat where i found it kinda ok
            '''
            self.SpawnVegetation()
            self.MoveMooses()
            self.MoveWolfs()
            self.MooseEat()
            self.MoosesGiveBirth()
            self.WolfsEat()
            self.WolfsGiveBirth()
            self.AnimalDies()
            self.IncreaseHunger()

            # Hunter Update
            counter += 1
            if counter == self.huntingLength:
                if self.hunting:
                    self.numberOfHunters = self.initialNumberOfHunters
                    self.InitializeHunters()
                    self.hunting = False
                else:
                    self.RemoveHunters()
                    self.numberOfHunters = 0
                    self.hunting = True
                counter = 0
            self.MoveHunters()
            self.Hunting()

            #Calculate
            area = self.forrest.size
            self.UpdatePopulation()
            self.populationVector[0,tIteration] = self.numberOfMooses
            self.populationVector[1,tIteration] = self.numberOfWolfs
            self.populationVector[2,tIteration] = self.numberOfHunters
            f = tIteration
            if (self.populationVector[0,tIteration] <= 10 or
                self.populationVector[1,tIteration] <= 10
                or self.populationVector[0,tIteration]>900):
                break
        return f
        
            #Dynamic probabilities???
            #self.mooseOffspringProbability = self.initialMooseOffspringProbability*self.numberOfMooses/area
            #self.wolfDeathProbability = self.initialWolfDeathProbability*self.numberOfWolfs/area
            #self.wolfOffspringProbability = self.initialWolfOffspringProbability*(self.numberOfWolfs/area)*(self.numberOfMooses/area)
            #print(self.wolfOffspringProbability)
    #Plot population graph and grid while runing simulation
    def RealTimePlotting(self,tIteration,plotter):
            plotter.SetupAxes()
            plotter.PlotForrest(1,self.forrest)
            plotter.PlotPopulation(0,self.populationVector, self.timeVector, self.totalTime,tIteration)
        #Solution for colabs
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.1)


class Plotting():

    def __init__(self):
        self.fig, self.axes = plt.subplots(2,1,figsize=(8,12))
    #Define plotting functions
    def SetupAxes(self):
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[0].set_box_aspect(1/3)
        self.axes[1].set_box_aspect(1)

    def PlotForrest(self,plotNumber,forrest):
        #Plot mooses
        (x,y) = np.where(forrest==1)
        self.axes[plotNumber].scatter(x,y, color='r')
        #Plot wolfs
        (x,y) = np.where(forrest==2)
        self.axes[plotNumber].scatter(x,y, color='k')
        #Plot Hunters
        (x,y) = np.where(forrest==3)
        self.axes[plotNumber].scatter(x,y, color='b')
        #General settings
        self.axes[plotNumber].axis('off')
        self.axes[plotNumber].axis('equal')

    def PlotPopulation(self,plotNumber, populationVector, timeVector, totalTime, tIter):
        #Plot graph
        self.axes[plotNumber].plot(timeVector[0:tIter],populationVector[0,0:tIter],'r-', label='Mooses')
        self.axes[plotNumber].plot(timeVector[0:tIter],populationVector[1,0:tIter],'k-', label='Wolfs')
        self.axes[plotNumber].plot(timeVector[0:tIter],populationVector[2,0:tIter],'b-', label='Hunters')
        #General settings
        self.axes[plotNumber].set_xlim(0,totalTime)
        self.axes[plotNumber].set_title('Population')
        self.axes[plotNumber].set_xlabel('Time [t]')
        self.axes[plotNumber].set_ylabel('Population [N]')
        self.axes[plotNumber].legend()
        self.axes[plotNumber].set_ylim(0,1.1*np.max(populationVector))
