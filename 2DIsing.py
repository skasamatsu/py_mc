import numpy as np
import random as rand

from mc import model, CanonicalMonteCarlo

class ising2D(model):
    '''This class defines the 2D ising model'''

    model_name = "ising2D"
    
    def __init__(self, J):
        self.J = J

    def energy(self,ising2D_config):
        ''' Calculate total energy of the 2D Ising model'''
        e = 0.0
        config = ising2D_config.config # config should be 2D numpy array
        for i in range(ising2D_config.lenX):
            for j in range(ising2D_config.lenY):
                e += self.J*config[i,j]*(config[i-1,j] + config[i,j-1])
        return e

    def trialstep(self, ising2D_config):
        # choose x,y randomly
        x = rand.randrange(ising2D_config.lenX)
        y = rand.randrange(ising2D_config.lenY)
        # Position of flipped spin to be used by newconfig():
        dconfig = [x,y]
        config = ising2D_config.config

        # Calculate energy change if spin flips at x,y
        x1 = x + 1
        if x == ising2D_config.lenX - 1:
            x1 = 0
        y1 = y + 1
        if y == ising2D_config.lenY - 1:
            y1 = 0
        dE = -2.0*self.J*config[x,y]*(config[x-1,y] + config[x1, y] + config[x, y-1] + config[x, y1])
        #print dconfig, dE
        return dconfig, dE

    def newconfig(self, ising2D_config, dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        ising2D_config.config[dconfig[0],dconfig[1]] *= -1
        return ising2D_config
        

        
class ising2D_config:
    '''This class defines the 2D Ising model configuration'''

    def __init__(self, lenX, lenY):
        self.lenX = lenX
        self.lenY = lenY
        self.config = np.empty([lenX, lenY])

    def prepare_random(self):
        for i in range(self.lenX):
            for j in range(self.lenY):
                if rand.random() >= 0.5:
                    self.config[i,j] = 1
                else:
                    self.config[i,j] = -1

    def __str__(self):
        s = ""
        for i in range(self.lenX):
            for j in range(self.lenY):
                if self.config[i,j] < 0:
                    s += "-"
                else:
                    s += "+"
            s += "\n"
        return s


if __name__ == "__main__":
    J = 1.0
    kT = abs(J) * 1.0
    config = ising2D_config(10,10)
    config.prepare_random()
    print config
    model = ising2D(J)
    calc = CanonicalMonteCarlo(model, kT, config)
    calc.run(1000000)
    print config
    
        
