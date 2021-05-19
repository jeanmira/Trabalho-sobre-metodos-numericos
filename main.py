# ------------------------------- /usr/bin/g++-7 ------------------------------#
# ------------------------------- coding: utf-8 -------------------------------#
# Criado por:   Jean Marcelo Mira Junior
#               Lucas Daniel dos Santos
# Versão: 1.0
# Criado em: 13/04/2021
# Sistema operacional: Linux - Ubuntu 20.04.1 LTS
# Python 3
# ------------------------------ Pacotes --------------------------------------#
import matplotlib.pyplot as plt
import biblioteca as bib
import numpy as np
# -----------------------------------------------------------------------------#


def f(s, a):
    fi, r, z = a
    bo = 0.4
    if(s != 0):
        return np.array([2-bo*z-np.sin(fi)/r, np.cos(fi), np.sin(fi)])
    else:
        return np.array([2-bo*z, np.cos(fi), np.sin(fi)])


# Método numérico de Euler
se, re = bib.edoEuler(f, (0, 0, 0), 0, 400, 0.01)

# Método numérico de Heun
sh, rh = bib.edoHeun(f, (0, 0, 0), 0, 400, 0.01)

# Método numérico de Runge-Kutta
sr, rr = bib.edoRungeKutta(f, (0, 0, 0), 0, 400, 0.01)

# Método numérico de Runge-Kutta-Fehlberg
sf, rf = bib.edoRungeKuttaFehlberg(
    f, (0, 0, 0), 0, 52, 0.01, 10 ** -3, 4, 0.1)

# bib.planilha(400, se, re, sh, rh, sr, rr, sf, rf)
bib.grafico(se, re, sh, rh, sr, rr, sf, rf)
# bib.gota(re, rh, rr, rf)
