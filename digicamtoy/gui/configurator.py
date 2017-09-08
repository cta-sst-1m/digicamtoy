from tkinter import *
from digicamtoy.core import tracegenerator
import inspect
#from tkinter.ttk import *


class ConfigurationWindow(Frame):

    """Notre fenêtre principale.

    Tous les widgets sont stockés comme attributs de cette fenêtre."""

    def __init__(self, fenetre, **kwargs):
        Frame.__init__(self, fenetre, width=768, height=576, **kwargs)

        self.pack(fill=BOTH)

        self.nb_clic = 0

        # Création de nos widgets

        self.message = Label(self, text="Vous n'avez pas cliqué sur le bouton.")

        self.message.pack()

        self.bouton_quitter = Button(self, text="Quitter", command=self.quit)

        self.bouton_quitter.pack(side="left")

        self.bouton_cliquer = Button(self, text="Cliquez ici", fg="red", command=self.cliquer)

        self.bouton_cliquer.pack(side="right")

    def cliquer(self):
        """Il y a eu un clic sur le bouton.



        On change la valeur du label message."""

        self.nb_clic += 1

        self.message["text"] = "Vous avez cliqué {} fois.".format(self.nb_clic)

        return

if __name__ == '__main__':

    a = inspect.signature(tracegenerator.TraceGenerator.__init__)
    window = Tk()

    #window.style = Style()
    #print(window.style.theme_names())
    # ('clam', 'alt', 'default', 'classic')
    #window.style.theme_use('clam')

    ConfigurationWindow(window)

    window.mainloop()

    window.destroy()