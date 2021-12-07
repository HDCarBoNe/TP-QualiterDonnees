#Imports librairies de base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import matplotlib.widgets as widgets
from matplotlib.widgets import Button
from matplotlib.widgets import Cursor
import math

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

def print_moy_min_max(climat):
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
    class SnaptoCursor(object):
        def __init__(self, ax, x, y):
            self.ax = ax
            self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
            self.marker, = ax.plot([0], [0], marker="o", color="crimson", zorder=3)
            self.x = x
            self.y = y
            self.txt = ax.text(0.7, 0.9, '')

        def mouse_move(self, event):
            if not event.inaxes: return
            x, y = event.xdata, event.ydata
            indx = np.searchsorted(self.x, [x])[0]
            x = self.x[indx]
            y = self.y[indx]
            self.ly.set_xdata(x)
            self.marker.set_data([x], [y])
            self.txt.set_text('Jours=%1.1f, Temp=%1.1f' % (x, y))
            self.txt.set_position((x, y))
            self.ax.figure.canvas.draw_idle()

    t = x_annuelle
    s = y_annuelle
    fig, ax = plt.subplots()

    # cursor = Cursor(ax)
    cursor = SnaptoCursor(ax, t, s)
    cid = plt.connect('motion_notify_event', cursor.mouse_move)

    ax.plot(t, s, )
    plt.axis([0, 365, -40, 40])
    plt.show()

def print_30j_glissant():
    x = x_annuelle
    y = y_annuelle
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.bar(x, y, align='center')

    ax2 = fig.add_subplot(212)
    ax2.bar(range(len(y)), y, align='center')
    plt.xticks(range(len(x)), x)

    plt.show()

####################################################################################
def print_climat_error():
    print(climat_error.sample(10))
####################################################################################

#Afficher les valeurs contenu dans le fichier excel
print_climat()
#Pour l’ échantillon SI, calculez : • moyenne par mois • écart type par mois • min /max par mois et par année
print_moy_min_max(climat)
#tracer les courbes de chaque mois avec une bibliothèque graphique python Matplotlib, 12 vues mensuelles
print_courbe_mois()
#Assembler les courbes sur un seul graphique (J1 -> J365) : vue annuelle
y_annuelle,x_annuelle = print_courbe_annuelle()
#Présenter la valeur lue en parcourant la courbe à l'aide du pointeur,
print_courbe_annuelle_pointeur()
#Présenter les valeurs précédentes par mois glissant de 30 jours centré sur la valeur lue
#print_30j_glissant()

########################################################################################################################
# Recommencez avec le jeu SI-erreur après avoir corrigé les valeurs en erreur. Précisez vos méthodes.
print_climat_error()
#print_moy_min_max(climat_error)

months = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre",
          "décembre"]
mean = []
std = []
minli = []
maxli = []
minyear = []
maxyear = []


def statistics(dataframe):
    for i, month in enumerate(months, start=0):
        print("\n\nStats du mois de " + months[i] + ":\n")
        print("moyenne du mois de " + months[i] + " : " + str(nanmean(dataframe[:, i])))
        print("Écart-type du mois de " + months[i] + " : " + str(nanstd(dataframe[:, i])))
        print("Min du mois de " + months[i] + " : " + str(nanmin(dataframe[:, i])))
        print("Max du mois de " + months[i] + " : " + str(nanmax(dataframe[:, i])))
        mean.append(nanmean(dataframe[:, i]))
        std.append(nanstd(dataframe[:, i]))
        minli.append(nanmin(dataframe[:, i]))
        maxli.append(nanmax(dataframe[:, i]))
    year_climat = dataframe[~isnan(dataframe)]
    minyear.append(nanmin(year_climat))
    maxyear.append(nanmax(year_climat))
    print('\n')
    print("Température minimale de l'année : " + str(minyear[0]))
    print("Température maximale de l'année : " + str(maxyear[0]))

def clean_data(dataframe):
    avg_month = nanmean(dataframe, axis=0)
    for i, avg_month in enumerate(avg_month, start=0):
        for j in range(len(dataframe)):
            if math.isnan(dataframe[j][i]):
                if j < 28:
                    dataframe[j][i] = round(avg_month)
            else:
                if 0 < j < 30:
                    if verif_data(dataframe[j - 1][i], dataframe[j][i], dataframe[j + 1][i]):
                        dataframe[j][i] = round(avg_month)
    np.savetxt("./Climat_cleaned.csv", dataframe, delimiter=";")


def verif_data(prev, data, next):
    moyenne = (prev + next) / 2
    if data > (moyenne + 7) or data < (moyenne - 7):
        return True
    return False

    np.savetxt("./Climat_cleaned.csv", dataframe, delimiter=";")


def calcul_rolling_mean(dataframe):
    df = pd.DataFrame(dataframe)
    roll = df.rolling(2, win_type='triang').sum()
    print("Rolling mean : " + roll)

def get_plot_from_dataframe():
    class Cursor(object):
        def __init__(self, ax, x, y):
            self.ax = ax
            self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
            self.marker, = ax.plot([1], [0], marker="o", color="crimson", zorder=3)
            self.x = x
            self.y = y
            self.txt = ax.text(0, 0, '')

        def mouse_move(self, event):
            if not event.inaxes: return
            x, y = event.xdata, event.ydata
            indx = searchsorted(self.x, [x])[0]
            x = self.x[indx]
            y = self.y[indx - 1]
            self.ly.set_xdata(x)
            self.marker.set_data([x], [y])
            self.txt.set_text(f'jour {x}, {y}°C')
            self.txt.set_color("crimson")
            self.txt.set_position((x + 1, y + 1))
            self.ax.figure.canvas.draw()

    def get_plot_from_dataframe(months, climat, mean, std, minli, maxli, minyear, maxyear):
        year_array = array([])
        for i, month in enumerate(months, start=0):
            year_array = append(year_array, climat[:, i])
        year_array = year_array[~isnan(year_array)]

        month = 0
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        title = plt.title(f"Température du mois de {months[month]} \n")
        suptitle = plt.suptitle("\n\nMoyenne: " + str(round(mean[month], 2)) + ", écart-type: " + str(
            round(std[month], 2)) + " min: " + str(round(minli[month], 2)) + " max: " + str(
            round(maxli[month], 2)), fontsize=10)
        plt.xlabel('Jours')
        plt.ylabel('Température')

        plt.axis([1, 31, nanmin(year_array) - 5, nanmax(year_array) + 5])
        plt.grid(True)
        plot, = plt.plot(range(1, 32, 1), climat[:, month], "b:o", lw=2)

        class Index:
            ind = 0

            def __init__(self, cursor):
                self.cursor = cursor

            def next(self, event):
                if self.ind < 11:
                    self.ind += 1
                    title.set_text(f"Température du mois de {months[self.ind]} \n ")
                    suptitle = plt.suptitle("\n\nMoyenne: " + str(round(mean[self.ind], 2)) + ", écart-type: " + str(
                        round(std[self.ind], 2)) + " min: " + str(round(minli[self.ind], 2)) + " max: " + str(
                        round(maxli[self.ind], 2)), fontsize=10)
                    plot.set_ydata(climat[:, self.ind])
                    self.cursor.x = arange(len(climat[:, self.ind]))
                    self.cursor.y = climat[:, self.ind]
                    plt.draw()
                elif self.ind == 11:
                    self.ind += 1
                    title.set_text(f"Température sur l'année \n ")
                    suptitle = plt.suptitle("\n\nmin: " + str(minyear[0]) + " max: " + str(maxyear[0]), fontsize=10)
                    x = [x for x in arange(365) if x % 30 == 0]
                    ax.set_xlim(right=365)
                    ax.set_xticks(x)
                    ax.set_xticklabels(x)
                    plot.set_xdata(range(0, len(year_array), 1))
                    plot.set_ydata(year_array)
                    self.cursor.x = arange(1, 366)
                    self.cursor.y = year_array
                    plt.draw()

            def prev(self, event):
                if self.ind > 0:
                    self.ind -= 1
                    title.set_text(f"Température du mois de {months[self.ind]} \n ")
                    suptitle = plt.suptitle("\n\nMoyenne: " + str(round(mean[self.ind], 2)) + ", écart-type: " + str(
                        round(std[self.ind], 2)) + " min: " + str(round(minli[self.ind], 2)) + " max: " + str(
                        round(maxli[self.ind], 2)), fontsize=10)
                    plot.set_ydata(climat[:, self.ind])
                    self.cursor.x = arange(len(climat[:, self.ind]))
                    self.cursor.y = climat[:, self.ind]
                    if self.ind == 11:
                        x = [x for x in arange(31) if x % 5 == 0]
                        ax.set_xlim(right=32)
                        ax.set_xticks(x)
                        ax.set_xticklabels(x)
                        plot.set_xdata(range(1, 32, 1))
                    plt.draw()

        days = arange(0, 31)
        celsius_degree = climat[days, 0]
        cursor = Cursor(ax, days, celsius_degree)
        plt.connect('motion_notify_event', cursor.mouse_move)

        callback = Index(cursor)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)

        plt.show()

climat_error = genfromtxt('./Climat-erreur.csv', delimiter=';', dtype=float, skip_header=True)
statistics(climat_error)
cleaned_climat = clean_data(climat_error)
#calcul_rolling_mean(climat_error)
#statistics(cleaned_climat)
#get_plot_from_dataframe(months, cleaned_climat, mean, std, minli, maxli, minyear, maxyear)