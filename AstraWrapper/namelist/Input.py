
import random

class Input:
    """class Input created as part of AstraWrapper to control the input beam and it's properties"""
    dictionary = {"FNAME": '',
                  "Add": 'F',
                  "N_add": 0,
                  "Ipart": 500,
                  "Species": 'electrons',
                  "Probe": 'F',
                  "Noise_reduc":'F',
                  "High_res": 'F',
                  "Q_total": '1.0',

                  "Ref_Ekin": 500,  #MeV
                  "Dist_pz": 'u',
                  "sig_Ekin": 1000, #keV
                  "Dist_z": 'u',    
                  "sig_z": 0,       #mm
                  "Dist_x":'G',
                  "sig_x": 1E-7,    #mm
                  "Lx": 0,          #mm
                  "x_off": 0,       #mm
                  "Dist_y":'G',
                  "sig_y": 1E-7,    #mm
                  "y_off": 0,       #mm
                  "Dist_px":'G',
                  "sig_px": 1E-7,   #eV
                  "Dist_py":'G',
                  "sig_py": 1E-7,   #eV
    }

    particleIdx = {'electrons': 1,
                   'positrons': 2,
                   'protons': 3,
                   'hydrogen': 4
    }

    # parameters for generating particles at source
    statusFlag = 5      #status of a particle that belongs to the beam
    massElectronInEv = 5.1E+5 
    isGaussian = ['G', 'g', 'gaussian', "Gaussian", "'g'", "'G'", "'gaussian'", "'Gaussian'"]
    isUniform = ['u', 'U', 'uniform', 'Uniform', "'u'", "'U'","'Uniform'", "'uniform'"]

    def __init__(self, filename):
        
        if isinstance(filename, str):
            if '.ini' in filename:
                self.filename = filename
            else:
                self.filename = filename + '.ini'
            self.dictionary['FNAME'] = filename
        else:
            raise ValueError(f"Class Input is expecting filename as a string.")


        
    def Gauss(self,sig, mu): #returns random number according to normal distribution
        return math.ceil(random.gauss(mu,sig))

    def Uniform(self,a, b, sig, mu): #returns random number uniformly between a and b, or according to mu, and sig of uniform distribution
        if a != None and b != None:
            return math.ceil(random.uniform(a,b))
        else:
            return math.ceil(random.uniform(mu - math.sqrt(3)*sig, mu + math.sqrt(3)*sig ))


    def generatePointSource(self, Pz = None , a_Px = None, b_Px = None, a_Py = None, b_Py = None):
        # function that randomly generates beam according to specified distributions and parameters, saves the distribution to an .ini file specifically 
        # for Astra. It is different than Astra generator because it creates a true point source and keeps Pz constant instead of the kinetic energy of 
        # a beam particle. 
        # arguments for this function are used for uniform distribution of spread of px or py only. For spread in px, py input values in eV

        if Pz is None:
            Pz = self.dictionary['Ref_Ekin']*1000000

        output = ' 0 0 0 0 0 ' + str(Pz + self.massElectronInEv) + ' 0 ' + str(self.dictionary['Q_total']) + '   ' + str(self.particleIdx[self.dictionary['Species']]) + '   ' + str(self.statusFlag) + '\n'

        for i in range(self.dictionary['Ipart'] - 1):

            px, py = 0, 0
            if self.dictionary['Dist_px'] in self.isGaussian:
                px = self.Gauss(self.dictionary['sig_px'], 0)
            elif self.dictionary['Dist_py'] in self.isUniform:
                px = self.Uniform(sig=self.dictionary['sig_px'], mu=0, a=a_Px, b=b_Px)
            else:
                raise ValueError(f"For px distribution, method generatePointSource() of class Generator is expecting a gaussian or uniform distribution.")

            if self.dictionary['Dist_py'] in self.isGaussian:
                py = self.Gauss(self.dictionary['sig_py'], 0)
            elif self.dictionary['Dist_py'] in self.isUniform:
                py = self.Uniform(sig=self.dictionary['sig_py'], mu=0, a=a_Py, b=b_Py )

            output += f" {xOffset} {yOffset} 0 {px} {py} 0 0 {self.dictionary['Q_total']}   {self.particleIdx[self.dictionary['Species']]}   {self.statusFlag}\n"
        
        with open(self.fileName + ".ini", 'w') as file:
            file.write(output)


    def generateSource(self, Pz = None , a_Px = None, b_Px = None, a_Py = None, b_Py = None, a_x = None, b_x = None, a_y = None, b_y = None):
        # function that randomly generates beam according to specified distributions and parameters, saves the distribution to an .ini file specifically 
        # for Astra. It is different than Astra generator because it keeps Pz constant instead of the kinetic energy of a beam particle. 
        # arguments for this function are used for uniform distribution of source size or the spread of px or py only. For spread in px, py input values in eV,
        # for source size in x,y use mm
        
        if Pz is None:
            Pz = self.dictionary['Ref_Ekin']*1000000

        output = ' 0 0 0 0 0 ' + str(Pz + self.massElectronInEv) + ' 0 ' + str(self.dictionary['Q_total']) + '   ' + str(self.particleIdx[self.dictionary['Species']]) + '   ' + str(self.statusFlag) + '\n'

        for i in range(nPart - 1):

            px, py, x, y = 0, 0, 0, 0

            # X
            if self.dictionary['Dist_x'] in self.isGaussian:
                x = self.Gauss(self.dictionary['sig_x'],self.dictionary['x_off'])
            elif self.dictionary["Dist_x"] in self.isUniform:
                x = self.Uniform(sig=self.dictionary['sig_x'], mu=self.dictionary['x_off'], a=a_x, b=b_x)
            else:
                raise ValueError(f"For x distribution, method generateSource() of class Generator is expecting a gaussian or uniform distribution.")

            # Y
            if self.dictionary['Dist_y'] in self.isGaussian:
                y = self.Gauss(self.dictionary['sig_y'],self.dictionary['y_off'])
            elif self.dictionary["Dist_y"] in self.isUniform:
                y = self.Uniform(sig=self.dictionary['sig_y'], mu=self.dictionary['y_off'], a=a_y, b=b_y)
            else:
                raise ValueError(f"For y distribution, method generateSource() of class Generator is expecting a gaussian or uniform distribution.")
            
            # Px
            if self.dictionary['Dist_px'] in self.isGaussian:
                px = self.Gauss(self.dictionary['sig_px'], 0)
            elif self.dictionary['Dist_py'] in self.isUniform:
                px = self.Uniform(sig=self.dictionary['sig_px'], mu=0, a=a_Px, b=b_Px)
            else:
                raise ValueError(f"For px distribution, method generatePointSource() of class Generator is expecting a gaussian or uniform distribution.")

            # Py
            if self.dictionary['Dist_py'] in self.isGaussian:
                py = self.Gauss(self.dictionary['sig_py'], 0)
            elif self.dictionary['Dist_py'] in self.isUniform:
                py = self.Uniform(sig=self.dictionary['sig_py'], mu=0, a=a_Py, b=b_Py )


                output += f" {x/1000} {y/1000} {0} {px} {py} {0} {0} {self.dictionary['Q_total']}   {self.particleIdx[self.dictionary['Species']]}   {self.statusFlag}\n"
        
        with open(self.fileName + ".ini", 'w') as file:
            file.write(output)

    def getNParticles(self):
        return self.dictionary['Ipart']
    def getSpecies(self):
        return self.dictionary['Species']
    def getProbe(self):
        return self.dictionary['Probe']
    def getCharge(self):
        return self.dictionary['Q_total']
    def getRefEkin(self):
        return self.dictionary['Ref_Ekin']
    def getDistributionX(self):
        return self.dictionary['Dist_x']
    def getDistributionY(self):
        return self.dictionary['Dist_y']
    def getDistributionZ(self):
        return self.dictionary['Dist_z']
    def getDistributionPx(self):
        return self.dictionary['Dist_px']
    def getDistributionPy(self):
        return self.dictionary['Dist_py']
    def getDistributionPz(self):
        return self.dictionary['Dist_pz']
    def getSigmaX(self):
        return self.dictionary['sig_x']
    def getSigmaY(self):
        return self.dictionary['sig_y']
    def getSigmaZ(self):
        return self.dictionary['sig_z']
    def getSigmaPx(self):
        return self.dictionary['sig_px']
    def getSigmaPy(self):
        return self.dictionary['sig_py']
    def getSigmaPz(self):
        return self.dictionary['sig_Ekin']
    def getXOff(self):
        return self.dictionary['x_off']
    def getYOff(self):
        return self.dictionary['y_off']
    def getText(self, idx):
        text = ''
        for key, val in self.dictionary.items():
            text += f"{key}({idx}) = {val}\n"
        return text

    def setNParticles(self, val):
        self.dictionary['Ipart'] = val
    def setSpecies(self,val):
        self.dictionary['Species'] = val
    def setProbe(self, val):
        self.dictionary['Probe'] = val
    def setCharge(self,val):
        self.dictionary['Q_total'] = val
    def setRefEkin(self,val):
        self.dictionary['Ref_Ekin'] = val
    def setDistributionX(self, val):
        self.dictionary['Dist_x'] = val
    def setDistributionY(self, val):
        self.dictionary['Dist_y'] = val
    def setDistributionZ(self, val):
        self.dictionary['Dist_z'] = val
    def setDistributionPx(self, val):
        self.dictionary['Dist_px'] = val
    def setDistributionPy(self,val):
        self.dictionary['Dist_py'] = val
    def setDistributionPz(self,val):
        self.dictionary['Dist_pz'] = val
    def setSigmaX(self,val):
        self.dictionary['sig_x'] = val
    def setSigmaY(self,val):
        self.dictionary['sig_y'] = val
    def setSigmaZ(self,val):
        self.dictionary['sig_z'] = val
    def setSigmaPx(self,val):
        self.dictionary['sig_px'] = val
    def setSigmaPy(self,val):
        self.dictionary['sig_py'] = val
    def setSigmaPz(self,val):
        self.dictionary['sig_Ekin'] = val
    def setXOff(self,val):
        self.dictionary['x_off'] = val
    def setYOff(self,val):
        self.dictionary['y_off'] = val