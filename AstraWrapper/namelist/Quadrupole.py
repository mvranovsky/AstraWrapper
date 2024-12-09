
import numpy as np

class Quadrupole:
    """docstring for Quadrupole"""

    dictionary = {"Q_type": '',
                  "Q_grad": 0,
                  "Q_noscale": 'T',
                  "Q_length": 0,
                  "Q_smooth": 0,
                  "Q_bore": 0,
                  "Q_pos": 0,
                  "Q_xoff": 0,
                  "Q_yoff": 0,
                  "Q_xrot": 0,
                  "Q_yrot": 0,
                  "Q_zrot": 0
    }



    def __init__(self, filename = None ,gradient = None, length = None, radius = None):

        if length != None and gradient != None and radius != None:
            self.inputFromFile = False
            self.length = length
            self.gradient = gradient
            if radius == 0:
                self.radius = 1E-9
            else:
                self.radius = radius

            self.dictionary['Q_length'] = length
            self.dictionary['Q_grad'] = gradient
            self.dictionary["Q_bore"] = 2*radius

            del self.dictionary['Q_type']
        elif filename != None:

            if not isinstance(filename,str) or 'data' not in filename:
                raise ValueError(f"Class quadrupole is expecting filename to be a string including 'data'.")
            
            self.inputFromFile = False

            self.filename = filename
            self.dictionary['Q_type'] = filename
            del self.dictionary['Q_grad']
            del self.dictionary['Q_length']
            del self.dictionary['Q_bore']

    def getGradient(self):
        if not inputFromFile:
            return self.gradient
        else:
            raise ValueError(f"Quadrupole class initialized by input from file.")
    def getRadius(self):
        if not inputFromFile:
            return self.radius
        else:
            raise ValueError(f"Quadrupole class initialized by input from file.")
    def getLength(self):
        if not inputFromFile:
            return self.length
        else:
            raise ValueError(f"Quadrupole class initialized by input from file.")
    def getXRotation(self):
        return self.dictionary['Q_xrot']
    def getYRotation(self):
        return self.dictionary['Q_yrot']
    def getZRotation(self):
        return self.dictionary['Q_zrot']
    def getOffsetX(self):
        return self.dictionary['Q_xoff']
    def getOffsetY(self):
        return self.dictionary['Q_yoff']
    def getPosition(self):
    	return self.dictionary['Q_pos']
    def getSmooth(self):
        return self.dictionary['Q_smooth']

    def getText(self, idx):
        text = ''

        for key, val in self.dictionary.items():
            text += f"{key}({idx}) = {val}\n"

        return text

    def setGradient(self, grad):
        if inputFromFile:
            raise ValueError(f"Quadrupole class initialized by input from file.")
        else:
            self.gradient = gradient
            self.dictionary['Q_grad'] = gradient
    def setRadius(self, radius):
        if inputFromFile:
            raise ValueError(f"Quadrupole class initialized by input from file.")
        else:
            if radius == 0:
                self.radius = 1E-9
                self.dictionary['Q_bore'] = 1E-9
            else:
                self.radius = radius
                self.dictionary['Q_bore'] = 2*radius
    def setLength(self, l):
        if inputFromFile:
            raise ValueError(f"Quadrupole class initialized by input from file.")
        else:
            self.length = l
            self.dictionary['Q_length'] = l
    def setXRotation(self, val):
        self.dictionary['Q_xrot'] = val
    def setYRotation(self, val):
        self.dictionary['Q_yrot'] = val
    def setZRotation(self, val):
        self.dictionary['Q_zrot'] = val
    def setOffsetX(self, val):
        self.dictionary['Q_xoff'] = val
    def setOffsetY(self, val):
        self.dictionary['Q_yoff'] = val
    def setPosition(self, val):
    	self.dictionary['Q_pos'] = val
    def setSmooth(self, val):
        self.dictionary['Q_smooth'] = val
   

    def gradientFunction(self, z):

        z = np.atleast_1d(z)  
        grad = np.atleast_1d(self.gradient)

        fVal = grad / (
            (1 + np.exp(-2 * z / self.radius)) *
            (1 + np.exp(2 * (z - self.length) / self.radius) )
        )

        return fVal if fVal.size > 1 else fVal[0]

    def integrateGradient(self):
        # numerical integration of the gradient profile 
        Sum = 0
        if inputFromFile:
            z, g = [],[]

            with open(self.filename, "r") as file:
                lines = file.readlines()

                for line in lines:
                    z.append(float(line.split(' ')[0]))
                    g.append(float(line.split(' ')[1]))

            for i in range(len(z)-1):
                Sum += (g[i] + g[i+1])/(2*(z[i+1] - z[i]))

        else:
            lower = -3*self.radius
            upper = self.length + 3*self.radius
            step = (upper - lower)/100

            for i in range(100):
                z = lower + i*step
                Sum += (self.gradientFunction(z) + self.gradientFunction(z + step) )/(2*step)

        return Sum












        


