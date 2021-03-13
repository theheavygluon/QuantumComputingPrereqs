#Note that I do not take full credit for this script as it is a slight modification to a script which could be found here: https://github.com/FelixDesrochers/Numerov
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import animation

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def ModifyPotential(potential):
    

    potential = potential.replace('^','**')

    pot_list = potential.rsplit('|')
    for i in [ i for i in range(1,(len(pot_list)-1)*2) if i%2==1 ]:
        insertion = 'np.absolute(' if i%4 ==1 else ')'
        pot_list.insert(i,insertion)

    potential=''.join(pot_list)

    potential = potential.replace('cos','np.cos')
    potential = potential.replace('sin','np.sin')
    potential = potential.replace('tan','np.tan')

    return potential


def VerifySyntaxPotential(potential):
    

    i=0
    while i == 0:
        try:
            x=0
            eval(potential)
        except SyntaxError:
            potential = input('The potential\'s syntax is incorrect enter a new one: ')
            potential = ModifyPotential(potential)
        else:
            i=1

    return potential


def VerifyLimitsPotential(potential):
    

    i=1
    while i == 1:
        eval_pot = list()
        x=-100
        eval_pot.append(eval(potential))
        x=100
        eval_pot.append(eval(potential))
        eval_pot = np.array(eval_pot)
        x = 0

        if eval_pot[eval_pot < eval(potential)]:
            QuestionPotential = input('The potential doesn\'t seem to be correct. Are you it corresponds to a bound state (y/n)? ')

            if QuestionPotential == 'n':
                potential = input('Enter a new potential: ')
                potential = ModifyPotential(potential)
                potential = VerifySyntaxPotential(potential)
            elif QuestionPotential == 'y':
                i = 0

        else :
                i = 0

    return potential


def GetFirstEnergyGuess(PotentialArray):
    

    First_E_guess = PotentialArray.min() +  (1/500000) * (PotentialArray.mean() + PotentialArray.min())

    return First_E_guess


def VerifyConcavity(PotentialArray, First_E_guess):
    

    i = 1
    while i == 1:
        print('First Energy guess:', First_E_guess)
        index_min=list()
        index_max=list()

        try:
            for i in range(0,len(PotentialArray)-2):

                if PotentialArray[i] > First_E_guess and PotentialArray[i+1] < First_E_guess:
                    index_min.append(i)

                elif PotentialArray[i] < First_E_guess and PotentialArray[i+1] > First_E_guess:
                    index_max.append(i)

                elif PotentialArray[i] == First_E_guess:
                    if PotentialArray[i-1] > First_E_guess and PotentialArray[i+1] < First_E_guess:
                        index_min.append(i)

                    elif PotentialArray[i-1] < First_E_guess and PotentialArray[i+1] > First_E_guess:
                        index_max.append(i)

            print('index max: ',index_max)
            print('index_min: ',index_min)

            if (max(index_max) > max(index_min)) and (min(index_max) > min(index_min)):
                concavity = 'positive'
            else:
                concavity = 'negative'

        except ValueError:
            First_E_guess = First_E_guess/2

        else:
            i = 0

    return concavity,First_E_guess


def EvaluateOnePotential(position,potential):
    

    x = position
    EvalPotential = eval(potential)

    return EvalPotential

def TranslationPotential(PositionPotential, PotentialArray):
    
    trans_y = PotentialArray.min()
    #index = float(np.where(PotentialArray==trans_y)[0])

    #trans_x = x_min + (Div * index)
    #trans_x = PositionPotential[index]

    PotentialArray = PotentialArray - trans_y
    #PositionPotential = PositionPotential - trans_x

    #print('trans_x; ',trans_x)
    print('trans_y; ',trans_y)

    return PositionPotential, PotentialArray

def TranslatePotential(potential,trans_x,trans_y):
    #x translation
    #potential = potential.replace('x','(x+' + str(trans_x) + ')')

    #y translation
    potential = potential + '-' +  str(trans_y)

    print(potential)

    return potential




def E_Guess(EnergyLevelFound, E_guess_try, iteration, First_E_guess):
    

    print('Iteration: ',iteration)

    if iteration == 1:
        E_guess = First_E_guess  
        return E_guess

    Lvl_found = list(EnergyLevelFound.keys())
    Lvl_found.sort()
    E_level_missing = [index for index,Energy in enumerate(Lvl_found) if not Energy <= index]
    if not E_level_missing:
        if not Lvl_found:
            E_level_guess = 0
        else:
            E_level_guess = max(Lvl_found) +1
    else:
        E_level_guess = min(E_level_missing)

    try:
        E_level_smaller = max([ E for E in E_guess_try.keys() if E <= E_level_guess ])
    except ValueError:
        E_level_smaller = None
    try:
        E_level_bigger = min([ E for E in E_guess_try.keys() if E > E_level_guess ])
    except ValueError:
        E_level_bigger = None

    if (not E_level_smaller == None) and (not E_level_bigger ==None):
        E_guess = ( E_guess_try[E_level_smaller][1] + E_guess_try[E_level_bigger][0] ) / 2

    elif not E_level_bigger == None:
        E_guess = E_guess_try[E_level_bigger][0]/2

    elif not E_level_smaller == None:
        E_guess = E_guess_try[E_level_smaller][1] * 2

    print('E_level_guess:', E_level_guess )
    print('E_level_bigger: ', E_level_bigger)
    print('E_level_smaller: ', E_level_smaller)

    return E_guess



def MeetingPointsPotential(E_guess, PotentialArray, PositionPotential, E_guess_try):
    

    p = 1
    iteration = 0
    end_program = False

    while p == 1:
        MeetingPoints = [None,None]
        for i in range(0,len(PotentialArray)-2):
            if (PotentialArray[i] < E_guess and PotentialArray[i+1] > E_guess) or (PotentialArray[i] > E_guess and PotentialArray[i+1] < E_guess) or (PotentialArray[i] == E_guess):
                if (MeetingPoints[0] == None) or (PositionPotential[i] < MeetingPoints[0]):
                    print('index rencontre min: ',i)
                    MeetingPoints[0] = PositionPotential[i]
                elif (MeetingPoints[1] == None) or (PositionPotential[i] > MeetingPoints[1]):
                    MeetingPoints[1] = PositionPotential[i]
                    print('index renccontre max: ', i)

        if (MeetingPoints[0] == None) or (MeetingPoints[1] == None):
            print('Restting the energy guess!\n')
            E_guess = (E_guess + max([k for j,k in E_guess_try.values() if k < E_guess]))/2
            iteration += 1
            print('E_guess: ',E_guess)
            if iteration > 10:
                end_program = True
                break
        else:
            p = 0
            MeetingPoints = tuple(MeetingPoints)

    return MeetingPoints,end_program,E_guess


def DetermineMinAndMax(MeetingPoints):
    

    Position_min = MeetingPoints[0] - (MeetingPoints[1] - MeetingPoints[0])/1
    Position_max =  MeetingPoints[1] + (MeetingPoints[1] - MeetingPoints[0])/1

    return Position_min,Position_max



def WaveFunctionNumerov(potential, E_guess, nbr_division, Initial_augmentation, Position_min, Position_max):
    

    WaveFunction = []

    Division = (Position_max - Position_min) / nbr_division

    WaveFunction.append((float(Position_min),0))
    WaveFunction.append((float(Position_min+Division), Initial_augmentation))

    index = 0
    PositionArray = np.arange(Position_min, Position_max, Division)

    for i in np.arange(Position_min + (2 * Division), Position_max, Division):
        x = i
        V_plus1 = eval(potential)

        x = PositionArray[index+1]
        V = eval(potential)

        x = PositionArray[index]
        V_minus1 = eval(potential)

        k_2_plus1 = 2 * (E_guess - V_plus1)
        k_2 = 2 * (E_guess - V)
        k_2_minus1 = 2 * (E_guess - V_minus1)

        psi = ((2 * (1 - (5/12) * (Division**2) * (k_2)) * (WaveFunction[-1][1])) - (1 + (1/12) * (Division**2) * k_2_minus1 ) * (WaveFunction[-2][1])) / (1 + (1/12) * (Division**2) * k_2_plus1)

        WaveFunction.append((i,psi))

        index += 1

    return WaveFunction



def NumberNodes(WaveFunction):
  

    NumberOfNodes = 0
    PositionNodes = list()

    for i in range(1,len(WaveFunction)-1):
        if (WaveFunction[i][1] > 0 and WaveFunction[i+1][1] < 0) or (WaveFunction[i][1] < 0 and WaveFunction[i+1][1] > 0) or (WaveFunction[i][1] == 0):
            NumberOfNodes += 1
            PositionNodes.append(WaveFunction[i][0])


    x = list()
    for position,wave in WaveFunction:
        x.append(position)
    x_max = max(x)

    return NumberOfNodes,PositionNodes,x_max



def VerifyTolerance(WaveFunction, Tolerance, E_guess, E_guess_try, NumberOfNodes):
    

    VerificationTolerance = 'yes' if np.absolute(WaveFunction[-1][1]) < Tolerance else 'no'
    print('Last value Wave Function: ', WaveFunction[-1][1])

    try:
        E_minus = E_guess_try[NumberOfNodes][1]
        E_plus = E_guess_try[NumberOfNodes + 1][0]
    except KeyError:
        pass
    else:
        if (E_guess < E_plus and E_guess > E_minus) and ((E_minus/E_plus) > 0.9999999999) :
            VerificationTolerance = 'yes'

    return VerificationTolerance

def CorrectNodeNumber(NumberOfNodes, PositionNodes, x_max, E_guess, E_guess_try):


    NumberOfNodesCorrected = NumberOfNodes
    try:
        if (E_guess_try[NumberOfNodes][1] > E_guess) and (E_guess_try[NumberOfNodes - 1][1] < E_guess):
            NumberOfNodesCorrected -= 1
    except KeyError:
        if (PositionNodes/x_max) > 94:
            NumberOfNodesCorrected -= 1

    return NumberOfNodesCorrected



def SaveEnergy(NumberOfNodes, E_guess, E_guess_try):
    

    try:
        E_guess_try[NumberOfNodes]

    except KeyError:
        E_guess_try[NumberOfNodes] = [E_guess, E_guess]
        return E_guess_try

    if E_guess < E_guess_try[NumberOfNodes][0]:
        E_guess_try[NumberOfNodes][0] = E_guess

    elif E_guess > E_guess_try[NumberOfNodes][1]:
        E_guess_try[NumberOfNodes][1] = E_guess

    return E_guess_try



def OuputEnergy(EnergyLevelFound):
    for i,Energy in EnergyLevelFound.items():
        print('Energy level', i, ':', Energy)


def DefineWhatToPlot(WaveFunctionFound, EnergyLevelFound):
    

    y_max = 1.1*EnergyLevelFound[max(EnergyLevelFound)]
    Y_by_E_level = (y_max/(max(EnergyLevelFound)+2))

    WavPlot = []
    for i in WaveFunctionFound.keys():
        x=[]
        y=[]
        for j in range(400,len(WaveFunctionFound[i])-240):
            if not (j > 3750 and np.absolute(WaveFunctionFound[i][j][1]) > (max(y)*0.07)):
                x.append(WaveFunctionFound[i][j][0])
                y.append(WaveFunctionFound[i][j][1])
        x = np.array(x)
        y = np.array(y)

        mult = (0.9 * Y_by_E_level)/(2 * y.max())
        y = (mult * y) + (Y_by_E_level * (i+1))
        WavPlot.append((x,y))

    min_x = x.min()
    max_x = x.max()

    WavLines = []
    for i in WaveFunctionFound.keys():
        Wav_line_y=[]
        for j in range(len(x)):
            Wav_line_y.append(Y_by_E_level * (i+1))
        WavLines.append((x,Wav_line_y))


    EnergyLines = []
    for i in WaveFunctionFound.keys():
        En_y = []
        for j in range(len(x)):
            En_y.append(EnergyLevelFound[i])
        EnergyLines.append((x,En_y))

    return y_max, min_x, max_x, WavPlot, WavLines, EnergyLines


def DrawWaveFunction(y_max, min_x, max_x, WavPlot, WavLines, EnergyLines, PositionPotential, PotentialArray):
    
  

    f,(En,Wav) = plt.subplots(1,2,sharey=True)


    f.suptitle("Schrödinger equation solutions",fontsize=20,fontweight='bold')



    lines = [Wav.plot(x,y,'b',label=r"$Re(\psi(x))$",zorder=3)[0] for x,y in WavPlot]
    lines2 =  [Wav.plot(x,y,'m',label=r"$Im(\psi(x))$",zorder=3)[0] for x,y in WavPlot]

    for x,y in WavLines:
        Wav.plot(x,y,'k--',zorder=1)


    Wav.axis([min_x, max_x, 0, y_max])


    Wav.plot(PositionPotential, PotentialArray, 'r',label='Potential',zorder=2)


    i = 0
    for x,y in EnergyLines:
        PlotColor = cm.viridis(i/len(EnergyLines))
        En.plot(x,y,'--',color=PlotColor,label='E'+str(i),zorder=2)
        i+=1


    En.axis([min_x, max_x, 0, y_max])

    En.plot(PositionPotential, PotentialArray, 'r',label='Potential',zorder=1)


    Wav.set_xlabel(r'x ($a_0$)')
    Wav.set_title('Wave Function',fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    leg1 = Wav.legend(newHandles, newLabels, loc='upper left', fontsize='x-small')
    leg1.get_frame().set_alpha(1)


    for i in range(len(EnergyLines)):
        Wav.text(((max_x - min_x) * 0.04) + min_x, WavLines[i][1][0] - (0.25 * (y_max/(len(EnergyLines)+2))), r'$\Psi_{%s}(x)$'%(i))


    En.set_xlabel(r'x ($a_0$)')
    En.set_ylabel('Energy (Hartree)')
    En.set_title('Energy levels',fontsize=14)
    leg2 = En.legend(loc='upper left', fontsize='x-small')
    leg2.get_frame().set_alpha(1)


    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def UpdateData(t):
        for j,line in enumerate(lines):
            x = WavPlot[j][0]
            y = ((WavPlot[j][1] - (WavLines[j][1][0]))  * np.cos(EnergyLines[j][1][0]*t/20)) + (WavLines[j][1][0])
            line.set_data(x,y)
        for j,line in enumerate(lines2):
            x = WavPlot[j][0]
            y = ((WavPlot[j][1] - (WavLines[j][1][0]))  * np.sin(EnergyLines[j][1][0]*t/20)) + (WavLines[j][1][0])
            line.set_data(x,y)

        return lines,lines2

    anim = animation.FuncAnimation(f, UpdateData, init_func=init, interval=10, blit=False, repeat=True, save_count=300, )


    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)

    plt.show()

    #Saving the animation
    #anim.save('Schrod.gif', writer='imagemagick', dpi=100, fps=25)




def Numerov(potential,E_level):
    import numpy as np




    x_V_min = -13
    x_V_max = 13
    nbr_division_V = 800000

    nbr_division = 5000

    Initial_augmentation = 0.00001

    Tolerance = 0.00000001





    E_lvl = list(range(0,E_level))



    i=1
    while i == 1:
        potential = ModifyPotential(potential)

        potential = VerifySyntaxPotential(potential)

        potential = VerifyLimitsPotential(potential)

        EvaluatePotential = np.vectorize(EvaluateOnePotential)
        DivisionPotential = (x_V_max - x_V_min) / nbr_division_V
        PositionPotential = np.arange(x_V_min,x_V_max,DivisionPotential)

        PotentialArray = EvaluatePotential(PositionPotential,potential)


        PositionPotential,PotentialArray = TranslationPotential(PositionPotential, PotentialArray)

        First_E_guess = GetFirstEnergyGuess(PotentialArray)

        concavity,First_E_guess = VerifyConcavity(PotentialArray, First_E_guess)

        if concavity == 'positive':
            i = 0

        elif concavity == 'negative':
            potential2 = input('The concavity of the potential isn\'t correct enter a new one (or "O" to overule): ')
            if potential2 == 'O':
                i = 0
            else :
                potential = potential2


    EnergyLevelFound = {} 
    WaveFunctionFound = {} 
    E_guess_try = {} 
    iteration = 1 
    E_found = list() 
    

    while not E_found == list(range(E_level)):

 

        E_guess = E_Guess(EnergyLevelFound,E_guess_try,iteration, First_E_guess)
        print('E_guess: ', E_guess)


        MeetingPoints,end_program,E_guess = MeetingPointsPotential(E_guess, PotentialArray, PositionPotential, E_guess_try)

        if end_program:
            break

        Position_min,Position_max = DetermineMinAndMax(MeetingPoints)


        WaveFunction = WaveFunctionNumerov(potential, E_guess, nbr_division, Initial_augmentation, Position_min, Position_max)


        NumberOfNodes,PositionNodes,x_max = NumberNodes(WaveFunction)
        print('NumberOfNodes:', NumberOfNodes)


        VerificationTolerance = VerifyTolerance(WaveFunction,Tolerance,E_guess,E_guess_try,NumberOfNodes)

        if VerificationTolerance == 'yes':
            print('Niveau d\'energie trouve!!\n\n')
            NumberOfNodesCorrected = CorrectNodeNumber(NumberOfNodes,PositionNodes,x_max,E_guess,E_guess_try)
            EnergyLevelFound.update({NumberOfNodesCorrected:E_guess})
            WaveFunctionFound.update({NumberOfNodesCorrected:WaveFunction})


        E_guess_try = SaveEnergy(NumberOfNodes, E_guess, E_guess_try)
        print('E_guess_try: ',E_guess_try)


        print('iterations:',iteration,'\n')
        iteration += 1

        E_found = list()
        for i in EnergyLevelFound.keys():
            E_found.append(i)
        E_found.sort()
        print('Energy level found',EnergyLevelFound)

    a = OuputEnergy(EnergyLevelFound)
    y_max,min_x,max_x,WavPlot,WavLines,EnergyLines = DefineWhatToPlot(WaveFunctionFound,EnergyLevelFound)
    b = DrawWaveFunction(y_max, min_x, max_x, WavPlot, WavLines, EnergyLines, PositionPotential, PotentialArray)

    return a, b 

#######################################################################################################################################

def eigs(potential,E_level, showWork=False):
    import numpy as np




    x_V_min = -13
    x_V_max = 13
    nbr_division_V = 800000

    nbr_division = 5000

    Initial_augmentation = 0.00001

    Tolerance = 0.00000001


    if showWork==False:
        blockPrint()
    else:
        adam = "cool"


    E_lvl = list(range(0,E_level))



    i=1
    while i == 1:
        potential = ModifyPotential(potential)

        potential = VerifySyntaxPotential(potential)

        potential = VerifyLimitsPotential(potential)

        EvaluatePotential = np.vectorize(EvaluateOnePotential)
        DivisionPotential = (x_V_max - x_V_min) / nbr_division_V
        PositionPotential = np.arange(x_V_min,x_V_max,DivisionPotential)

        PotentialArray = EvaluatePotential(PositionPotential,potential)


        PositionPotential,PotentialArray = TranslationPotential(PositionPotential, PotentialArray)

        First_E_guess = GetFirstEnergyGuess(PotentialArray)

        concavity,First_E_guess = VerifyConcavity(PotentialArray, First_E_guess)

        if concavity == 'positive':
            i = 0

        elif concavity == 'negative':
            potential2 = input('The concavity of the potential isn\'t correct enter a new one (or "O" to overule): ')
            if potential2 == 'O':
                i = 0
            else :
                potential = potential2


    EnergyLevelFound = {} 
    WaveFunctionFound = {} 
    E_guess_try = {} 
    iteration = 1 
    E_found = list() 
    

    while not E_found == list(range(E_level)):

 

        E_guess = E_Guess(EnergyLevelFound,E_guess_try,iteration, First_E_guess)
        print('E_guess: ', E_guess)


        MeetingPoints,end_program,E_guess = MeetingPointsPotential(E_guess, PotentialArray, PositionPotential, E_guess_try)

        if end_program:
            break

        Position_min,Position_max = DetermineMinAndMax(MeetingPoints)


        WaveFunction = WaveFunctionNumerov(potential, E_guess, nbr_division, Initial_augmentation, Position_min, Position_max)


        NumberOfNodes,PositionNodes,x_max = NumberNodes(WaveFunction)
        print('NumberOfNodes:', NumberOfNodes)


        VerificationTolerance = VerifyTolerance(WaveFunction,Tolerance,E_guess,E_guess_try,NumberOfNodes)

        if VerificationTolerance == 'yes':
            print('Niveau d\'energie trouve!!\n\n')
            NumberOfNodesCorrected = CorrectNodeNumber(NumberOfNodes,PositionNodes,x_max,E_guess,E_guess_try)
            EnergyLevelFound.update({NumberOfNodesCorrected:E_guess})
            WaveFunctionFound.update({NumberOfNodesCorrected:WaveFunction})


        E_guess_try = SaveEnergy(NumberOfNodes, E_guess, E_guess_try)
        print('E_guess_try: ',E_guess_try)


        print('iterations:',iteration,'\n')
        iteration += 1

        E_found = list()
        for i in EnergyLevelFound.keys():
            E_found.append(i)
        E_found.sort()
        print('Energy level found',EnergyLevelFound)

    if showWork==False:
        enablePrint()
    else:
        adam = "cool"

    return list(EnergyLevelFound.values())
