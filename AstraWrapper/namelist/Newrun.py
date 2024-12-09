

class Newrun:
    """class Newrun used as wrapper around namelist Newrun used in Astra."""


    dictionary = {"RUN": 1,
                  "Distribution": '',
                  "Xrms": -1,
                  "Yrms": -1,
                  "Track_All": 'T',
                  "check_ref_part": 'F',
                  "PHASE_SCAN": 'F',
                  "H_max": 0.001,
                  "H_min": 0,
                  "Max_step": 1000000
    }

    isTrue = ['T', "'T'", 'True', 'true', "'true'", "'True'", 't', "'t'"]


    def __init__(self,filename = None):

        if filename is not None and isinstance(filename, str):
            self.dictionary['Distribution'] = filename
            self.filename = filename


    def getRun(self):
        return self.dictionary['RUN']
    def getDistribution(self):
        return self.dictionary['Distribution']
    def getXrms(self):
        return self.dictionary['Xrms']
    def getYrms(self):
        return self.dictionary['Yrms']
    def isTrackingAll(self):
        if self.dictionary['Track_All'] in self.isTrue:
            return True
        else:
            return False
    def isCheckingReferenceParticle(self):
        if self.dictionary['check_ref_part'] in self.isTrue:
            return True
        else:
            return False
    def isPhaseScan(self):
        if self.dictionary['PHASE_SCAN'] in self.isTrue:
            return True
        else:
            return False
    def getHmax(self):
        return self.dictionary['H_max']
    def getHmin(self):
        return self.dictionary['H_min']
    def getMaximumSteps(self):
        return self.dictionary['Max_step']
    def getText(self, idx):
        text = ''
        for key, val in self.dictionary.items():
            text += f"{key}({idx}) = {val}\n"
        return text


    def setRun(self,val):
        self.dictionary['RUN'] = val
    def setDistribution(self,val):
        self.dictionary['Distribution'] = val
    def setXrms(self,val):
        self.dictionary['Xrms'] = val
    def setYrms(self,val):
        self.dictionary['Yrms'] = val
    def setTrackingAll(self,val):
        if isinstance(val, bool):
            if val:
                self.dictionary['Track_All'] = 'T'
            else:
                self.dictionary['Track_All'] = 'F'
        else:
            self.dictionary['Track_All'] = val

    def setCheckingReferenceParticle(self,val):
        if isinstance(val, bool):
            if val:
                self.dictionary['check_ref_part'] = 'T'
            else:
                self.dictionary['check_ref_part'] = 'F'
        else:
            self.dictionary['check_ref_part'] = val
    def setPhaseScan(self,val):
        if isinstance(val, bool):
            if val:
                self.dictionary['PHASE_SCAN'] = 'T'
            else:
                self.dictionary['PHASE_SCAN'] = 'F'
        else:
            self.dictionary['PHASE_SCAN'] = val
    def setHmax(self,val):
        self.dictionary['H_max'] = val
    def setHmin(self,val):
        self.dictionary['H_min'] = val
    def setMaximumSteps(self,val):
        self.dictionary['Max_step'] = val


		