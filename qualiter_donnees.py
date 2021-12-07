#Imports librairies de base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import matplotlib.widgets as widgets
from matplotlib.widgets import Button
from matplotlib.widgets import Cursor

#Bibliothèques pour Cross validation


#Affichage de Graphes
from IPython.display import Markdown, display, HTML
from numpy import genfromtxt, nanmean, isnan, nanstd, nanmin, nanmax, searchsorted, arange, array, append
#from graph import get_plot_from_dataframe
c_file='Climat.xlsx'
climat_file = pd.ExcelFile(c_file)

climat = pd.read_excel(climat_file, climat_file.sheet_names[0])
climat_error = pd.read_excel(climat_file, climat_file.sheet_names[1])
mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre",
            "décembre"]
moyenne = []
std = []
mintemp = []
maxtemp = []
min_an = []
max_an = []
x_annuelle = np.array([])
y_annuelle = np.array([])

def print_climat():
    print(climat.sample(10))
    print(climat_error.sample(10))

def print_moy_min_max():
    for i, month in enumerate(mois, start=0):
        print("\n\nStats du mois de " + mois[i] + ":\n")
        print("moyenne du mois de " + mois[i] + " : " + str(nanmean(climat.iloc[:, i])))
        print("Écart-type du mois de " + mois[i] + " : " + str(nanstd(climat.iloc[:, i])))
        print("Min du mois de " + mois[i] + " : " + str(nanmin(climat.iloc[:, i])))
        print("Max du mois de " + mois[i] + " : " + str(nanmax(climat.iloc[:, i])))
        moyenne.append(nanmean(climat.iloc[:, i]))
        std.append(nanstd(climat.iloc[:, i]))
        mintemp.append(nanmin(climat.iloc[:, i]))
        maxtemp.append(nanmax(climat.iloc[:, i]))

    an_climat = climat[~isnan(climat)]
    min_an.append(nanmin(an_climat))
    max_an.append(nanmax(an_climat))
    print('\n')
    print("Température minimale de l'année : " + str(min_an[0]))
    print("Température maximale de l'année : " + str(max_an[0]))

def print_courbe_mois():
    for i in range(12):
        title = plt.title("Température du mois de %s\n"%mois[i])
        plt.xlabel('Jours')
        plt.ylabel('Température')
        x = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30])
        y = np.array([])
        for j in range(0, 31):
            y = np.append(y, climat.iloc[:,i][j])
        plt.plot(x, y, 'b')
        plt.axis([1, 31, min_an[0] - 5, max_an[0] + 5])
        plt.grid(True)
        plt.show()

def print_courbe_annuelle():
    title = plt.title("Température annuelle")
    plt.xlabel('Jours')
    plt.ylabel('Température')
    x = np.array([])
    for day in range(0, 365):
        x = np.append(x, day)
        x_annuelle = np.append(x, day)
    # x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    y = np.array([])
    for i in range(12):
        for j in range(0, len(climat.iloc[:,i])):
            if not isnan(climat.iloc[:,i][j]):
                y = np.append(y, climat.iloc[:,i][j])
                y_annuelle = np.append(y, climat.iloc[:,i][j])

    plt.plot(x, y, 'b')
    plt.axis([1, 365, min_an[0] - 5, max_an[0] + 5])
    plt.grid(True)
    plt.show()
    return y_annuelle,x_annuelle

def print_courbe_annuelle_pointeur():
    # x and y arrays for definining an initial function
    #x = np.linspace(0, 10, 100)
    #y = np.exp(x ** 0.5) * np.sin(5 * x)
    x = x_annuelle
    y = y_annuelle
    # Plotting
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x, y, color='b')
    ax.grid()
    # Defining the cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)
    # Creating an annotating box
    annot = ax.annotate("", xy=(0, 0), xytext=(-40, 40), textcoords="offset points",
                        bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                        arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False)
    # Function for storing and showing the clicked values
    coord = []

    def onclick(event):
        global coord
        coord.append((event.xdata, event.ydata))
        x = event.xdata
        y = event.ydata

        # printing the values of the selected point
        print([x, y])
        annot.xy = (x, y)
        text = "({:.2g}, {:.2g})".format(x, y)
        annot.set_text(text)
        annot.set_visible(True)
        fig.canvas.draw()  # redraw the figure

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


#Afficher les valeurs contenu dans le fichier excel
print_climat()
#Pour l’ échantillon SI, calculez : • moyenne par mois • écart type par mois • min /max par mois et par année
print_moy_min_max()
#tracer les courbes de chaque mois avec une bibliothèque graphique python Matplotlib, 12 vues mensuelles
#print_courbe_mois()
#Assembler les courbes sur un seul graphique (J1 -> J365) : vue annuelle
y_annuelle,x_annuelle = print_courbe_annuelle()
#Présenter la valeur lue en parcourant la courbe à l'aide du pointeur,
print_courbe_annuelle_pointeur()
