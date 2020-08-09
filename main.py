# -*- coding: utf-8 -*-
from cnn2d import train_and_predict_3cnn2d
from cnn1d import train_and_predict_4cnn1d
from ensemble import ensemble_all

if __name__ == '__main__':
    print("model 3cnn2d training....")
    train_and_predict_3cnn2d()
    print("model 4cnn1d training....")
    train_and_predict_4cnn1d()
    print("ensemble_all...")
    ensemble_all()
