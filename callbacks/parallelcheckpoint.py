import os
import keras
from keras.callbacks import ModelCheckpoint, Callback


"""
- ModelCheckpoint for parallel computing
- Usage:
    - checkpoint = ParallelModelCheckpoint(model, filepath='./cifar10_resnet_ckpt.h5', 
    monitor='val_acc', verbose=1, save_best_only=True) # 解决多GPU运行下保存模型报错的问题
"""
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, outputPath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        filepath = os.path.sep([outputPath, "model.hdf5"])  # need to add epoch
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, \
                verbose, save_best_only, save_weights_only, mode, period)
        self.single_model = model

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


"""
- ParallelCheckpoint
- Usage:
    - cbk = ParallelCheckpoint(original_model)
      parallel_model.fit(..., callbacks=[cbk])
"""
class ParallelCheckpoint(Callback):
    def __init__(self, model, outputPath, every=5, startAt=0):
        # call the parent constructor
        super(Callback, self).__init__()
        self.model_to_save = model
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs=None):
        # check to see if the model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath, "epoch_{}.hdf5".format(self.intEpoch + 1)])
            self.model_to_save.save(p, overwrite=True)
        # increment the internal epoch counter
        self.intEpoch += 1







