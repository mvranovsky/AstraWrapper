import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Circle


class Cavity:
    """Class Cavity to wrap namelist cavity for Astra. Part of AstraWrapper module. For now, only used as a generator and an input for 
    for field map of a quadrupole magnet. """

    dictionary = {"File_Efield": '',
                  "C_noscale": 'T',
                  "C_pos": 0,         # m
                  "C_smooth": 0,
                  "Com_grid": 'T',
                  "Ex_stat" : 'T',
                  "Bx_stat" : "T",
                  "C_xoff" : 0,       # m 
                  "C_yoff": 0,        # m
                  "C_xrot": 0,        # rad
                  "C_yrot": 0,        # rad
                  "C_zrot": 0         # rad
    }

    isTrue = ['T', "'T'", 'True', 'true', "'true'", "'True'", 't', "'t'"]

    # lists of values for z position and gradient values
    z_val = None
    G_val = None

    # parameters for wobbles of gradient function
    gradAmp = 1    #T/m
    gradFreq = 100
    gradIniPhase = 0


    # parameters for wobbles of skew angle
    skewAmp1 = 1    #degrees
    skewFreq1 = 10
    skewIniPhase1 = 0
    skewAmp2 = 0.5  #degrees
    skewFreq2 = 30
    skewIniPhase2 = 0.05 


    # parameters for wobbles of magnetic centre x
    MCXAmp1 = 0.001     #m
    MCXFreq1 = 30
    MCXIniPhase1 = 0
    MCXAmp2 = 0.002     #m
    MCXFreq2 = 5
    MCXIniPhase2 = 0.005


    # parameters for wobbles of magnetic centre x
    MCYAmp1 = 0.001     #m
    MCYFreq1 = 10
    MCYIniPhase1 = 0
    MCYAmp2 = 0.003     #m
    MCYFreq2 = 5
    MCYIniPhase2 = 0.005


    def __init__(self, filename ):

        if not isinstance(filename, str):
            raise ValueError('argument filename is expected a string- either the name of the existing name of files with field map, or the name where the generated field map will be saved. ')

        self.filename = filename.split('.')[0]

        self.dictionary['File_Efield'] = self.filename


    # getters
    def getFilename(self):
        return self.dictionary["File_Efield"]
    def getPosition(self):
        return self.dictionary['C_pos']
    def isNoscale(self):
        if self.dictionary['C_noscale'] in self.isTrue:
            return True
        else:
            return False
    def getSmooth(self):
        return self.dictionary['C_smooth']
    def isCommonGrid(self):
        if self.dictionary["Com_grid"] in self.isTrue:
            return True
        else:
            return False
    def isStatic(self):
        if self.dictionary['Bx_stat'] in self.isTrue or self.dictionary['Ex_stat'] in self.isTrue:
            return True
        else:
            return False
    def getOffsetX(self):
        return self.dictionary['C_xoff']
    def getOffsetY(self):
        return self.dictionary['C_yoff']
    def getXRotation(self):
        return self.dictionary['C_xrot']
    def getYRotation(self):
        return self.dictionary['C_yrot']
    def getZRotation(self):
        return self.dictionary['C_zrot']
    def getText(self, idx):
        text = ''
        for key, val in self.dictionary.items():
            text += f"{key}({idx}) = {val}\n"
        return text
    def getGradientAmplitude(self):
        return self.gradAmp
    def getGradientFrequency(self):
        return self.gradFreq
    def getGradientInitialPhase(self):
        return self.gradIniPhase
    def getSkewAngleAmplitude1(self):
        return self.skewAmp1
    def getSkewAngleFrequency1(self):
        return self.skewFreq1
    def getSkewAngleInitialPhase1(self):
        return self.skewIniPhase1
    def getSkewAngleAmplitude2(self):
        return self.skewAmp2
    def getSkewAngleFrequency2(self):
        return self.skewFreq2
    def getSkewAngleInitialPhase2(self):
        return self.skewIniPhase2
    def getMagneticCentreXAmplitude1(self):
        return self.MCXAmp1
    def getMagneticCentreXFrequency1(self):
        return self.MCXFreq1
    def getMagneticCentreXInitialPhase1(self):
        return self.MCXIniPhase1
    def getMagneticCentreXAmplitude2(self):
        return self.MCXAmp2
    def getMagneticCentreXFrequency2(self):
        return self.MCXFreq2
    def getMagneticCentreXInitialPhase2(self):
        return self.MCXIniPhase2
    def getMagneticCentreYAmplitude1(self):
        return self.MCXAmp1
    def getMagneticCentreYFrequency1(self):
        return self.MCXFreq1
    def getMagneticCentreYInitialPhase1(self):
        return self.MCXIniPhase1
    def getMagneticCentreYAmplitude2(self):
        return self.MCXAmp2
    def getMagneticCentreYFrequency2(self):
        return self.MCXFreq2
    def getMagneticCentreYInitialPhase2(self):
        return self.MCXIniPhase2



    # setters
    def setPosition(self, val):
        self.dictionary['C_pos'] = val
    def setNoscale(self, val):
        if isinstance(val, bool):
            if val:
                self.dictionary['C_noscale'] = 'T'
            else:
                self.dictionary['C_noscale'] = 'F'
        else:
            self.dictionary['C_noscale'] = val
    def setSmooth(self,val):
        self.dictionary['C_smooth'] = val
    def setCommonGrid(self, val):
        if isinstance(val, bool):
            if val:
                self.dictionary['Com_grid'] = 'T'
            else:
                self.dictionary['Com_grid'] = 'F'
        else:
            self.dictionary['Com_grid'] = val
    def setStatic(self, val):
        if isinstance(val, bool):
            if val:
                self.dictionary['Bx_stat'] = 'T'
                self.dictionary['Ex_stat'] = 'T'
            else:
                self.dictionary['Bx_stat'] = 'T'
                self.dictionary['Ex_stat'] = 'T'
        else:
            self.dictionary['Bx_stat'] = 'T'
            self.dictionary['Ex_stat'] = 'T'
    def setOffsetX(self, val):
        self.dictionary['C_xoff'] = val
    def getOffsetY(self, val):
        self.dictionary['C_yoff'] = val
    def getXRotation(self, val):
        self.dictionary['C_xrot'] = val
    def getYRotation(self, val):
        self.dictionary['C_yrot'] = val
    def getZRotation(self, val):
        self.dictionary['C_zrot'] = val
    def setGradientAmplitude(self,val):
        self.gradAmp = val
    def setGradientFrequency(self,val):
        self.gradFreq = val
    def setGradientInitialPhase(self, val):
        self.gradIniPhase = val
    def setSkewAngleAmplitude1(self, val):
        self.skewAmp1 = val
    def setSkewAngleFrequency1(self,val):
        self.skewFreq1 = val
    def setSkewAngleInitialPhase1(self,val):
        self.skewIniPhase1 = val
    def setSkewAngleAmplitude2(self, val):
        self.skewAmp2 = val
    def setSkewAngleFrequency2(self,val):
        self.skewFreq2 = val
    def setSkewAngleInitialPhase2(self,val):
        self.skewIniPhase2 = val
    def setMagneticCentreXAmplitude1(self,val):
        self.MCXAmp1 = val
    def setMagneticCentreXFrequency1(self,val):
        self.MCXFreq1 = val
    def setMagneticCentreXInitialPhase1(self,val):
        self.MCXIniPhase1 = val
    def setMagneticCentreXAmplitude2(self,val):
        self.MCXAmp2 = val
    def setMagneticCentreXFrequency2(self,val):
        self.MCXFreq2 = val
    def setMagneticCentreXInitialPhase2(self,val):
        self.MCXIniPhase2 = val
    def setMagneticCentreYAmplitude1(self,val):
        self.MCXAmp1 = val
    def setMagneticCentreYFrequency1(self,val):
        self.MCXFreq1 = val
    def setMagneticCentreYInitialPhase1(self,val):
        self.MCXIniPhase1 = val
    def setMagneticCentreYAmplitude2(self,val):
        self.MCXAmp2 = val
    def setMagneticCentreYFrequency2(self,val):
        self.MCXFreq2 = val
    def setMagneticCentreYInitialPhase2(self,val):
        self.MCXIniPhase2 = val



    def integrateGradProfile(self, z_val = None, G_val = None, showPlot = False):
        #equidistant integration- just linear, because ASTRA linearly interpolates between data points

        if z_val is None and G_val is None:
            z_val = self.z_val
            G_val = self.G_val


        if len(z_val) != len(G_val):
            print("Ranges of z and G are not equal.")
            return
        

        sum1 = 0
        for i in range(len(G_val) - 1):
            #gradient in T/mm
            sum1 += (( G_val[i+1] + G_val[i] )/2)*( z_val[i+1] - z_val[i]) 
        
        if showPlot:
            plt.plot(z_val,G_val,color='blue',label='Gradient')
            plt.title('Gradient of the field [T/mm] ')
            plt.xlabel("z [mm]")
            plt.legend(loc='best')
            plt.grid()
            plt.show()
        

        return sum1

    def gradFunction0(self, z, wobbles):

        fVal = 0
        if z >= 0 and z <= self.qLength:
            fVal = z*(self.grad2 - self.grad1 )/self.qLength + self.grad1
            if wobbles:
                fVal += self.gradAmp*math.sin(self.gradFreq*z + self.gradIniPhase)

        return fVal


    def radiusFunction(self, z):
        if z >= 0 and z <= self.qLength:
            return ( z*(self.radius2- self.radius1)/self.qLength + self.radius1 )
        else:
            return 3


    def gradFunction(self, z, wobbles):
        # Ensure z is a NumPy array (this handles both scalar and array inputs)
        z = np.atleast_1d(z)  # Converts scalar to 1-element array if necessary

        # Initialize grad as an array of zeros, same shape as z
        grad = np.zeros_like(z)

        # Apply conditions vectorized (element-wise)
        grad[(z >= 0) & (z <= self.qLength)] = (
            z[(z >= 0) & (z <= self.qLength)] * (self.grad2 - self.grad1) / self.qLength + self.grad1
        )
        grad[z < 0] = self.grad1
        grad[z > self.qLength] = self.grad2

        # Calculate fVal using vectorized operations
        fVal = grad / (
            (1 + np.exp(-2 * z / self.radius1)) *
            (1 + np.exp(2 * (z - self.qLength) / self.radius2) )
        )

        # Handle wobbles condition vectorized
        if wobbles:
            fVal += self.gradAmp * np.sin(self.gradFreq * z + self.gradIniPhase)

        # If input was a scalar, return a scalar
        return fVal if fVal.size > 1 else fVal[0]


    def generateGradProfile(self, qLength, BTipField ,Qbore1 = None, Qbore2 = None,xFocusing = True,grad1 = None, grad2 = None, wobbles = False, fieldType = 1, nPoints = 100, fileOutputName = None, showPlot = False):
        #function to generate gradient profile, input either gradient or radius at start and end positions
        #(will be linearly interpolated) and the BtipField. Other parameters will be calculated according 
        #to BTipField = Grad*R_Q. One can specify the type of field, name of the file with output or number 
        #of points where function will be evaluated
        
        self.qLength = qLength
        self.BTipField = BTipField
        Zpos, gradVal, radius = [],[],[]
        if Qbore1 != None and Qbore2 != None:
            print("Will be generating field according to bore radius input and the tip field.")
            self.radius1 = Qbore1
            self.radius2 = Qbore2
            self.grad1 = BTipField/Qbore1
            self.grad2 = BTipField/Qbore2
            Zpos, gradVal, radius = self.gradient(BTipField ,fieldType, nPoints, wobbles, False)
            if not xFocusing:
                for i,g  in enumerate(gradVal):
                    gradVal[i] = -g
        elif grad1 != None and grad2 != None:
            print("Will be generating field according to gradient input and the tip field.")
            self.radius1 = BTipField/math.fabs(grad1)
            self.radius2 = BTipField/math.fabs(grad2)
            self.grad1 = grad1
            self.grad2 = grad2
            Zpos, gradVal, radius = self.gradient(BTipField ,fieldType, nPoints, wobbles, True)
        else:
            raise ValueError("BTipField has to be set at all times and then either bore radius or gradients are set at the beginning and the end.")


        if fileOutputName != None:
            self.quadName = fileOutputName

        if showPlot:
            self.plotGandR(Zpos, gradVal, radius)

        self.z_val = Zpos
        self.G_val = gradVal

        profileG = ''
        for i in range(len(Zpos) ):
            profileG += f"{Zpos[i]} {gradVal[i]}\n"

        apertureR = ''

        apertureR = f"0 {self.radius1*1000}\n{self.qLength} {self.radius2*1000}\n"


        # save the radius to aperture/quadName
        if ".dat" in self.quadName:
            with open("aperture/" + self.quadName, "w") as file:
                file.write(apertureR)
            with open(self.quadName, "w") as file:
                file.write(profileG)
        else:
            with open("aperture/" + self.quadName + ".dat", "w") as file:
                file.write(apertureR)
            with open(self.quadName + ".dat", "w") as file:
                file.write(profileG)

        return Zpos, gradVal, radius


    def gradient(self,BTipField, fieldType, nPoints, wobbles, gradSet):


        Zpos, gradVal, radius = [], [], []
        if fieldType == 0:
            for i in range(nPoints):
                z = i*self.qLength/nPoints
                Zpos.append( z )

                if gradSet:
                    g = self.grad1 + z*(self.grad2 - self.grad1)/self.qLength 
                    gradVal.append(g)
                    radius.append(BTipField/g )
                else:
                    r = (self.radius1 + (self.radius2 - self.radius1)*z/self.qLength)
                    gradVal.append( BTipField/r )
                    radius.append( r )
        elif fieldType == 1:
            dist = 5*self.radius1 + self.qLength + 5*self.radius1
            for i in range(math.ceil(nPoints*3/2 + 1) ):
                z = -5*self.radius1 + i*dist/(math.ceil(3*nPoints/2) )
                Zpos.append(z)
                if gradSet:
                    g = self.gradFunction(z, wobbles)
                    gradVal.append( g )
                    radius.append( self.radiusFunction(z) )
                else:
                    radius.append( self.radiusFunction(z) )  
                    r = (self.radius1 + (self.radius2 - self.radius1)*z/self.qLength)
                    g = BTipField/(r*( (1 + np.exp(-2 * z / self.radius1)) *(1 + np.exp(2 * (z - self.qLength) / self.radius2) ) ) )
                    gradVal.append( g )
        else:
            raise ValueError(f"fieldType {fieldType} is not implemented, only 0 for top hat field, 1 for astra generated gradients with fringe fields.")

        return Zpos, gradVal, radius

    def skewAngle(self, z, skewAngleWobbles):

        if skewAngleWobbles:
            if isinstance(z,float):
                return self.skewAmp1*math.sin(z*self.skewFreq1 + self.skewIniPhase1) + self.skewAmp2*math.sin(z*self.skewFreq2 + self.skewIniPhase2) 
            else:
                return self.skewAmp1*np.sin(z*self.skewFreq1 + self.skewIniPhase1) + self.skewAmp2*np.sin(z*self.skewFreq2 + self.skewIniPhase2) 

        else:
            if isinstance(z,float):
                return 0
            else:
                return 0*z


    def magCenterX(self,z, magCentreXWobbles):

        if magCentreXWobbles:
            if isinstance(z,float):
                return self.MCXAmp1*math.cos( z*self.MCXFreq1 + self.MCXIniPhase1 ) + self.MCXAmp2*math.sin( z*self.MCXFreq2 + self.MCXIniPhase2 ) 
            else:
                return self.MCXAmp1*np.cos( z*self.MCXFreq1 + self.MCXIniPhase1 ) + self.MCXAmp2*np.sin( z*self.MCXFreq2 + self.MCXIniPhase2 ) 
        else:
            if isinstance(z, float):
                return 0
            else:
                return 0*z

    def magCenterY(self,z, magCentreYWobbles):

        if magCentreYWobbles:
            if isinstance(z, float):
                return self.MCYAmp1*math.cos( z*self.MCYFreq1 + self.MCYIniPhase1 ) + self.MCYAmp2*math.sin( z*self.MCYFreq2 + self.MCYIniPhase2 ) 
            else:
                return self.MCYAmp1*np.cos( z*self.MCYFreq1 + self.MCYIniPhase1 ) + self.MCYAmp2*np.sin( z*self.MCYFreq2 + self.MCYIniPhase2 ) 
        else:
            if isinstance(z,float):
                return 0
            else: 
                return 0*z


    def generateFieldMap(self, qLength, BTipField ,Qbore1 = None, Qbore2 = None, xFocusing = True,grad1 = None, grad2 = None,gradWobbles = False,fieldType = 1, nGradPoints = 100,nFMPoints = 21, fileOutputName = None , magCentreXWobbles=False, magCentreYWobbles = False ,skewAngleWobbles = False, showPlot = True):

        Zpos, gradVal, radius = self.generateGradProfile(qLength, BTipField, Qbore1 = Qbore1, Qbore2=Qbore2,xFocusing=xFocusing, grad1 = grad1, grad2=grad2, wobbles=gradWobbles, fieldType=fieldType, nPoints = nGradPoints, fileOutputName=fileOutputName, showPlot = showPlot)

        self.radius = radius
        skewAng = list(self.skewAngle(np.array(Zpos), skewAngleWobbles))
        magCentreX = list(self.magCenterX(np.array(Zpos), magCentreXWobbles))
        magCentreY = list(self.magCenterY(np.array(Zpos), magCentreYWobbles))

        if showPlot:
            self.plotGenerationParameters(Zpos, skewAng, gradVal, magCentreX, magCentreY, radius)

        rad = max([self.radius1, self.radius2])

        # Define a grid
        x, y, z = np.linspace(-rad, rad, nFMPoints), np.linspace(-rad, rad, nFMPoints), np.array(Zpos)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.z = z
        self.x = x
        self.y = y

        grad = np.array(gradVal)
        gradVal_3D = grad[ np.newaxis, np.newaxis,:]  # Make gradVal 3D along z-axis

        Bx =( np.sin( (2*np.pi/360)*self.skewAngle(Z, skewAngleWobbles) )*gradVal_3D*(X - self.magCenterX(Z, magCentreXWobbles)) +
              np.cos( (2*np.pi/360)*self.skewAngle(Z, skewAngleWobbles) )*gradVal_3D*(Y - self.magCenterY(Z, magCentreYWobbles))  )
        By =( np.cos( (2*np.pi/360)*self.skewAngle(Z, skewAngleWobbles) )*gradVal_3D*(X - self.magCenterX(Z, magCentreXWobbles)) -
              np.sin( (2*np.pi/360)*self.skewAngle(Z, skewAngleWobbles) )*gradVal_3D*(Y - self.magCenterY(Z, magCentreYWobbles))  )


        # Calculate the derivative of Bx along z using central difference
        dz = np.diff(self.z)  # Differences between consecutive z values
        dBx_dz = np.zeros_like(Bx)

        # Central difference (for internal points, handling variable dz)
        for i in range(1, len(self.z) - 1):
            dBx_dz[:, :, i] = (Bx[:, :, i + 1] - Bx[:, :, i - 1]) / (self.z[i + 1] - self.z[i - 1])

        # Forward and backward difference for boundaries
        dBx_dz[:, :, 0] = (Bx[:, :, 1] - Bx[:, :, 0]) / (Zpos[1] - Zpos[0])            # Forward difference at the start
        dBx_dz[:, :, -1] = (Bx[:, :, -1] - Bx[:, :, -2]) / (Zpos[-1] - Zpos[-2])       # Backward difference at the end

        # Calculate Bz, ensuring no division by zero
        epsilon = 1e-9  # Small value to avoid singularities
        denominator = ((X - self.magCenterX(Z, magCentreXWobbles)) * np.sin(self.skewAngle(Z, skewAngleWobbles)) +
                       (Y - self.magCenterY(Z, magCentreYWobbles)) * np.cos(self.skewAngle(Z, skewAngleWobbles))) + epsilon

        Bz = dBx_dz * X * Y / denominator
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.current_field = Bx


        self.saveFieldMap(X,Y,Z,Bx,By,Bz)

        if not showPlot:
            return 

        max_abs_value = max(np.max(np.abs(Bx)), np.max(np.abs(By)))
        max_abs_value = 1


        # Create the figure and axis
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)  # Adjust space for slider and buttons
        self.ax.set_xlabel("x [mm]")
        self.ax.set_ylabel("y [mm]")

        # Initial plot: Show Bx at z-index 0
        z_index = 0
        #taking .T transposed matrix, because imshow() expects aruments y,x and not x,y like all other plotting functions
        self.field_plot = self.ax.imshow(Bx[:, :, z_index].T, origin='lower', cmap='viridis',extent=[-rad, rad, -rad, rad], vmin=-max_abs_value, vmax=max_abs_value )
        self.ax.set_title(f'Magnetic Field at z = {z_index} m')


        # Colorbar with label
        cbar = plt.colorbar(self.field_plot)  # Create color bar
        cbar.set_label('B (T)', rotation=270, labelpad=15)  # Add label with rotation


        # Slider Axes
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'z-position', z[0], z[-1], valinit=z[0], valfmt='%.3f')

        # Radio Button Axes for selecting Bx or By
        ax_radio = plt.axes([0.05, 0.2, 0.15, 0.2])
        radio = RadioButtons(ax_radio, ('Bx', 'By', 'Bz'))


        # Connect slider and radio button events to update functions
        self.slider.on_changed(self.update)
        radio.on_clicked(self.switch_field)

        self.circle = plt.Circle((0,0),radius[z_index] ,color='black', fill=False, linewidth=2)
        self.ax.add_patch(self.circle)

        plt.show()



    # Update function for the slider
    def update(self,val):
        # Find the closest z-index to the current slider value
        z_idx = (np.abs(self.z - val)).argmin()
        
        self.field_plot.set_data(self.current_field[:, :, z_idx].T)  # Update the plot data
        self.ax.set_title(f'Magnetic Field at z = {self.z[z_idx]:.3f}')  # Update title with actual z value
        self.circle = plt.Circle((0,0),self.radius[z_idx] ,color='black', fill=False, linewidth=2)
        #self.ax.add_patch(circle)

        self.fig.canvas.draw_idle()

    def switch_field(self,label):

        if label == 'Bx':
            self.current_field = self.Bx
        elif label == 'By':
            self.current_field = self.By
        elif label == 'Bz':
            self.current_field = self.Bz  # Add this line for Bz support
        
        self.update(self.slider.val)  # Redraw with new field


    def saveFieldMap(self, X,Y,Z,Bx, By, Bz):

        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError(f"The shapes of coordinates are not equal. ")
        if Bx.shape != By.shape or Bx.shape != Bz.shape:
            raise ValueError(f"The shapes of Bx, By, Bz fields are not equal. ")
        if Bx.shape != X.shape:
            raise ValueError(f"The shapes of fields and coordinates are not equal. Bx: {Bx.shape}, X: {X.shape}")


        outBx = str(self.x.size)
        for x in self.x:
            outBx += ' ' + str(x) 

        outBx += f"\n{self.y.size} "
        for y in self.y:
            outBx += f"{y} "

        outBx += f"\n{self.z.size} "
        for z in self.z:
            outBx += f"{z} "

        outBx += "\n"

        outBy = str(outBx)
        outBz = str(outBx)

        for k in range(self.z.size):
            for j in range(self.y.size):
                for i in range(self.x.size):
                    outBx += f"{Bx[i,j,k]} "
                    outBy += f"{By[i,j,k]} "
                    outBz += f"{Bz[i,j,k]} "
                outBx += "\n"
                outBy += "\n"
                outBz += "\n"

        with open("cavity/3D" + self.quadName + ".bx","w") as file:
            file.write(outBx)
  
        with open("cavity/3D" + self.quadName + ".by","w") as file:
            file.write(outBy)

        with open("cavity/3D" + self.quadName + ".bz","w") as file:
            file.write(outBz)


        
    def plotGenerationParameters(self,z_val, alpha, G, magCenterX, magCenterY , radius ):

        plt.figure(figsize=(5,10))
        row = 4; col =1
        plt.subplots_adjust(hspace =.4)
        
        
        plt.subplot(row,col,1)
        plt.title("Skew angle alpha")
        plt.plot(z_val,alpha,label='Skew angle', color = 'blue')
        plt.ylim(-2,2)
        plt.ylabel('Skew Angle [degrees] ')
        plt.xlabel("z [m]")
        plt.grid()
        
        plt.subplot(row,col,2)
        plt.title("gradient profile")
        plt.plot(z_val,G,'-', color='red')
        plt.xlabel("z [m]")
        plt.ylabel('Gradient of the field [T/m] ')
        plt.grid()
        
        plt.subplot(row,col,3)
        
        maxVal = math.ceil(max([math.fabs(num*1000) for num in magCenterX + magCenterY] ))

        plt.title("magnetic centre offset")
        plt.plot(z_val,[num*1000 for num in magCenterX],label='magnetic centre x', color = 'red')
        plt.plot(z_val,[num*1000 for num in magCenterY],label='magnetic centre y', color = 'blue')
        plt.xlabel("z [m]")
        plt.ylabel("offset [mm]")
        #plt.ylim(-maxVal, maxVal)
        plt.legend(loc='lower right')
        plt.grid()

        plt.subplot(row,col,4)


        radiusChosen = []
        zChosen = []
        for i,z in enumerate(z_val):
            if radius[i] <1:
                zChosen.append( float(z) )
                radiusChosen.append( float(radius[i]*1000) )


        plt.title("radius of aperture")
        plt.plot(zChosen,radiusChosen,label='aperture radius', color = 'blue')
        plt.xlabel("z [m]")
        plt.ylabel("radius [mm]")
        plt.ylim(0, math.ceil(max([self.radius1,self.radius2])*100 )*10 )
        plt.grid()


        plt.tight_layout()
        plt.show()


    def plotGandR(self, z_val, G, radius):

        plt.figure(figsize=(5,10))
        row = 2; col =1
        plt.subplots_adjust(hspace =.4)
        
        plt.subplot(row,col,1)
        plt.title("gradient profile")
        plt.plot(z_val,G,'-', color='red')
        plt.xlabel("z [m]")
        plt.ylabel('Gradient of the field [T/m] ')
        plt.grid()
        

        plt.subplot(row,col,2)

        radiusChosen = []
        zChosen = []
        for i,z in enumerate(z_val):
            if radius[i] <1:
                zChosen.append( float(z) )
                radiusChosen.append( float(radius[i]*1000) )


        plt.title("radius of aperture")
        plt.plot(zChosen,radiusChosen,label='aperture radius', color = 'blue')
        plt.xlabel("z [m]")
        plt.ylabel("radius [mm]")
        plt.ylim(0, math.ceil(max([self.radius1,self.radius2])*100 )*10 )
        plt.grid()


        plt.tight_layout()
        plt.show()
