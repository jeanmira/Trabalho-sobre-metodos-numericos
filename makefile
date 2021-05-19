# ------------------------------- /usr/bin/g++-7 ------------------------------#
# ------------------------------- coding: utf-8 -------------------------------#
# Criado por:   Jean Marcelo Mira Junior
# Versão: 1.0
# Criado em: 23/04/2021
# Sistema operacional: Linux - Ubuntu 20.04.1 LTS
# Python 3
# ------------------------------ Pacotes --------------------------------------#

all: packages run

packages:
	sudo apt-get update
	pip3 install numpy
	pip3 install matplotlib
	sudo apt-get update
	sudo apt-get upgrade

run:
	python3 main.py

