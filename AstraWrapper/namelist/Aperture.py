



class Aperture: 
    '''class used for AstraWrapper to introduce aperture. A single piece of aperture is defined with 1 class. It can be defined with 
       with name of input file which contains aperture definition as Astra is expecting it, or one can use arguments radius and length
       and the class will specify it in the input file for Astra or as an input file as cavity. 
    '''
    dictionary = {"File_Aperture": '',
                  "Ap_Z1": 0,
                  "Ap_Z2": 0,
                  "Ap_R": 0,
                  "A_pos": 0,
                  "A_xoff": 0,
                  "A_yoff": 0,
                  "A_xrot": 0,
                  "A_yrot": 0,
                  "A_zrot": 0
    }

    def __init__(self, filename = None, radius = None, length = None):

        if filename != None and '.dat' not in filename:
            filename = filename.split('.')[0] + '.dat'


        if  isinstance(filename, str) and  radius == None and length == None:
            self.filename = filename
            self.readExistingFile()
            self.dictionary['File_Aperture'] = filename
            del self.dictionary['Ap_Z1']
            del self.dictionary['Ap_Z2']
            del self.dictionary['Ap_R']
        elif filename == None and radius != None and length != None:
            self.filename = filename
            self.radius = radius
            self.length = length
            self.dictionary['Ap_Z1'] = 0
            self.dictionary['Ap_Z2'] = length
            self.dictionary['Ap_R'] = radius
            del self.dictionary['File_Aperture'] 
        elif filename != None and radius != None and length != None:
            self.filename = filename
            self.radius = radius
            self.length = length
            self.writeToFile()
            self.dictionary['File_Aperture'] = filename
            del self.dictionary['Ap_Z1']
            del self.dictionary['Ap_Z2']
            del self.dictionary['Ap_R']
        else:
            raise SyntaxError(f"Aperture class can be initialized either with an existing file with aperture is written as Astra expects it, or with radius and length.")

    def getPosition(self):
        return self.dictionary['Ap_pos']
    def getRadius(self):
        return self.radius
    def getLength(self):
        return self.length
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


    def setPosition(self, z):
        self.dictionary['Ap_pos'] = z
    def setRadius(self,radius):
        self.radius = radius
        if 'Ap_R' in self.dictionary:
            self.dictionary['Ap_R'] = radius
        else:
            self.writeToFile()
    def setLength(self, length):
        self.length = length
        if 'Ap_Z2' in self.dictionary:
            self.dictionary['Ap_Z2'] = length
        else:
            self.writeToFile()
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


    def getText(self, idx):
        text = ''

        for key, val in self.dictionary.items():
            text += f"{key}({idx}) = {val}\n"

        return text

    def readExistingFile(self):

        lines = []
        try:
            with open(self.filename, "r") as file:
                lines = file.readlines()
        except Exception as e:
            raise ValueError(f"Filename could not be found: {e}")


        firstVal = float(lines[0].split(" ")[0])
        if firstVal != 0:
            raise ValueError(f"The input file with aperture values expects to have 0 at the first z position. Value {firstVal} is not 0")


        for line in reversed(lines):
            if line.strip():
                self.length = float( line.split(" ")[0] )

        self.radius = float( lines[0].split(" ")[1] )


    def writeToFile(self):

        line = f"0 {self.radius}\n{self.length} {self.radius}"

        with open(self.filename, "w") as file:
            file.write(line)



