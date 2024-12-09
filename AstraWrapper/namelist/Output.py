

class Output:
    """ Class output to control output namelist for Astra Wrapper. """

    dictionary = {"ZSTART":0,     #m
                  "ZSTOP": 1,     #m
                  "Zemit": 100,
                  "Zphase": 1,
                  "Step_width":0,
                  "Step_max":0
                 }

    screenNum = 0

    def __init__(self, filename = None):
        if filename is not None and isinstance(filename, str):
            self.filename = filename

    # getters
    def getZStart(self):
        return self.dictionary['ZSTART']
    def getZStop(self):
        return self.dictionary['ZSTOP']
    def getZemit(self):
        return self.dictionary['Zemit']
    def getZPhase(self):
        return self.dictionary['Zphase']
    def getStepWidth(self):
        return self.dictionary['Step_width']
    def getStepMax(self):
        return self.dictionary['Step_max']
    def getText(self, idx):
        text = ''
        for key, val in self.dictionary.items():
            text += f"{key}({idx}) = {val}\n"
        return text


    # setters
    def setZStart(self,val):
        self.dictionary['ZSTART'] = val
    def setZStop(self,val):
        self.dictionary['ZSTOP'] = val
    def setZemit(self,val):
        self.dictionary['Zemit'] = val
    def setZPhase(self,val):
        self.dictionary['Zphase'] = val
    def setStepWidth(self,val):
        self.dictionary['Step_width'] = val
    def setStepMax(self,val):
        self.dictionary['Step_max'] = val

    def addScreen(self, position):
        self.dictionary[f"Screen({screenNum})"] = position
        screenNum += 1
    def removeScreen(self,position):
        for key, val in self.dictionary.items():
            if position == val:
                del self.dictionary[key]




		