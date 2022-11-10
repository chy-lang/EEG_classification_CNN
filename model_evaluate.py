import torch
import numpy as np
from RES_model import ComplexConvNeuralNetwork as model
classes = ['T0', 'T1', 'T2']
batch_size=32
def precision(matrix):
    precision_dict = {'T0': 0, 'T1': 0, 'T2': 0}
    for i in classes:
        true_predict_num = matrix[i][i]
        all_predict_num = 0
        for j in classes:
            all_predict_num = all_predict_num + matrix[j][i]
        precision_dict[i] = round(true_predict_num / all_predict_num, 4)
    return precision_dict

def accuracy(classes, matrix):
    right_num = 0
    sum_num = 0
    for row in classes:
        right_num += matrix[row][row]
        for column in classes:
            sum_num += matrix[row][column]
    return round(right_num / sum_num, 4)
