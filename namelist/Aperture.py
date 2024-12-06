



class Aperture: 

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

    def __init__(self, fileName = None, radius = None, length = None):

        if fileName != None and isinstance(fileName, str) and  radius == None and length == None:
            self.apertureName = fileName
            self.readExistingFile()
            self.dictionary['File_Aperture'] = fileName
            del self.dictionary['Ap_Z1']
            del self.dictionary['Ap_Z2']
            del self.dictionary['Ap_R']
        elif fileName == None and radius != None and length != None:
            self.apertureName = fileName
            self.radius = radius
            self.length = length
            self.dictionary['Ap_Z1'] = 0
            self.dictionary['Ap_Z2'] = length
            self.dictionary['Ap_R'] = radius
            del self.dictionary['File_Aperture'] 
        elif fileName != None and radius != None and length != None:
            self.apertureName = fileName
            self.radius = radius
            self.length = length
            self.writeToFile()
            self.dictionary['File_Aperture'] = fileName
            del self.dictionary['Ap_Z1']
            del self.dictionary['Ap_Z2']
            del self.dictionary['Ap_R']
        else:
            raise SyntaxError(f"Aperture class can be initialized either with an existing file with aperture is written as Astra expects it, or with radius and length.")

    def readExistingFile(self):

        lines = []
        with open(self.fileName, "r") as file:
            lines = file.readlines()


        firstVal = float(lines[0].split(" ")[0])
        if firstVal != 0:
            raise ValueError(f"The input file with aperture values expects to have 0 at the first z position. Value {firstVal} is not 0")


        for line in reversed(lines):
            if line.strip():
                self.length = float( line.split(" ")[0] )

        self.radius = float( lines[0].split(" ")[1] )


    def writeToFile(self):


        line = f"0 {self.radius}\n{self.length} {self.radius}"




        with open(self.fileName, "w") as file:


