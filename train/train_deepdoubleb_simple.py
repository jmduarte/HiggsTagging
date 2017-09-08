
import sys
import os
copypath = ['../models',
            '../generator',
            '../train']
for p in reversed(copypath):
    sys.path.insert(0, p)
print sys.path
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
keras.backend.set_image_data_format('channels_first')


class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from training_base import training_base
import sys

args = MyClass()
args.inputDataCollection = '/cms-sc17/convert_deepDoubleB_simple_train_val/dataCollection.dc'
args.outputDir = 'train_deep_simple/'

#also does all the parsing
train=training_base(testrun=True,args=args)


if not train.modelSet():
    from models import dense_model

    train.setModel(dense_model)
    
    train.compileModel(learningrate=0.0001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])
    

model,history,callbacks = train.trainModel(nepochs=10, 
                                 batchsize=128, 
                                 stop_patience=10, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)


