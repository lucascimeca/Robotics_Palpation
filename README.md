# Robotics Palpation (SoRo) - 25/06/2020
### Luca Scimeca (luca.scimeca@live.com) 

![](/assets/NMI_palpation_short.gif | width=100)

This repository contains code and data for the article "Action Augmentation of Soft Tactile Perception for Soft-Body Palpation", written under the supervision of Dr Fumiya Iida, and in collaboration with Dr Josie Hughes, Dr Thrishantha Nanayakkara and Mr Liang He, in the Engineering Department of the University of Cambridge.

## System Requirements

* The code was tested with PyCharm Professional 2019 and Anaconda

#### Dependencies
- [x] Python 3.6
- [x] matplotlib 3.0.3
- [x] numpy 1.16.4
- [x] pandas 0.25.1
- [x] scikit-learn
- [x] scipy

#### Installation (Approximately 20min set-up time)
- [x] Install github and clone the repository locally 
- [x] Install Anaconda and create a virtual env with Python 3.6
- [x] install required dependencies above
- [x] Install Pycharm and set the project interpreter to the newly created Anaconda environment

#### Demo & Instructions
- [x] In Pycharm, open a new project and select the repository folder
- [x] Set the 'src' folder as the source folder
- [x] Run 'main_bayesian_exploration.py' within the 'src' folder, to run experiments online
- [x] run 'generate_figures.py' to run the experiments with some pre-set hyperparameters and generate analytical figures automatically in the 'results' folder 

* A [video example](https://bitbucket.org/lucascimeca/nmi_palpation/raw/b0febc1b1a25e84360d0b80d630a3369fe926180/assets/code_demo.mp4 "download") of running the 'main_bayesian_exploration.py' is contained in the "assets" folder.

The tactile data is contained in the "data" folder. Each subfolder contains a different set of experiments. Please refer to the manuscript for further informations.
The remaining code includes robot control and experimental code with a UR5 robotic arm and CySkin sensor technology.