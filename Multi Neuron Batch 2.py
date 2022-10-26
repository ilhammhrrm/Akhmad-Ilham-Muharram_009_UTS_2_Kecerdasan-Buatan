# Multi Neuron Batch Input
#Inisialisasi nummpy
import numpy as np
# Inisialisasi Variabel Input
# 6 batcth Inputs setiap batch berisi 10 
inputs =[   
            # Inputs 1
            [2.0,-1.4,3.9,-2.15,4.0,7.1,5.2,6.4,7.5,-1.12],
            # Inputs 2
            [3.0,1.0,-1.1,3.1,5.1,4.12,3.3,4.0,5.5,-0.5],
            # Inputs 3
            [2.2,-1.14,2.1,0.9,0.45,5.12,-0.11,0.88,-3.55,3.1],
            # Inputs 4
            [6.1,-2.13,2.14,-5.1,6.2,1.1,2.09,0.7,-0.14,0.6],
            # Inputs 5
            [4.1,0.51,0.42,0.42,-0.52,0.31,0.31,-0.43,4.21,3.22],
            # Inputs 6
            [3.33,-4.13,1.53,2.1,5.1,6.12,2.41,-4.32,6.45,0.16]
        ]
# Inisialisasi Variabel Weights 1
weights_1 =[
            # Neuron 1
            [0.4,-0.8,0.1,0.2,0.5,2.0,0.7,-0.13,0.45,0.6],
            # Neuron 2
            [0.26,0.91,0.14,0.25,0.54,-0.52,0.21,0.82,0.31,0.42],
            # Neuron 3
            [0.51,0.44,-0.17,0.32,0.46,0.67,0.12,0.19,-0.34,0.71],
            # Neuron 4
            [0.2,0.41,0.3,-0.6,0.52,0.72,-0.35,0.66,-0.82,0.12],
            # Neuron 5
            [-0.14,0.8,-0.75,0.86,0.41,0.65,0.22,-0.6,0.7,0.55]
        ] 
# Inisialisasi bias 1
# Jumlah bias pada layer 1 berisi 5
biases_1 = [2.0,4.0,1.5,3.0,0.5]
# Inisialisasi Variabel Weights 2
# Jumlah neuron sesuai dengan jumlah bias pada layer ke 2, yaitu 3
# Di setiap neuron sesuai dengan jumlah bias pada layer 1, yaitu 5
Weights_2 =[
            # Neuron 1
            [1.2,2.2,1.3,2.1,0.2],
            # Neuron 2
            [1.4,2.5,0.6,2.4,0.5],
            # Neuron3 
            [2.7,1.8,0.9,2.4,2.1]
        ]     
# Inisialisasi bias 2
# Jumlah bias pada layer 2 berisi 3
biases_2 = [2.1,1.2,2.3]
# Perhitungan output layer 1 
layer_outputs_1 = np.dot(inputs, np.array(weights_1).T) + biases_1
# Perhitungan output layer 1 
layer_outputs_2 = np.dot(layer_outputs_1,np.array(Weights_2).T)+ biases_2
# Print layer_output 2
print(layer_outputs_2)
