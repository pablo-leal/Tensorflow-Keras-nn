__author__='Pablo Leal'

import argparse

from keras.callbacks import LambdaCallback

import trainer.board as board
import trainer.loader as loader
import trainer.modeller as modeller
import trainer.saver as saver

from trainer.constans import BATCH_SIZE, CHECKPOINT_PERIOD
from trainer.constans import EPOCHS
from trainer.constans import PREDICTION_LENGTH
from trainer.constans import WINDOW_LENGTH

def saveModelToCloud(epoch, period=1):
    if epoch % period = 0:
        server.saveModelToCloud(model, pathToJobDir + '/epochs_' + jobname, '{:03d}'.format(epoch))

if __name__='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-train-file',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '-job-name',
        help='GCS to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '-job-dir',
        help='GCS to write checkpoints and export models',
        required=True
    )
    args=parser.parse_args()
    arguments = args.__dict__

    pathToJobDir = arguments.pop('job_dir')
    jobName = arguments.pop('job_name')
    pathToData = arguments.pop('train_file')

    trainingDataDict, trainingLabelsDict, testingDataDict, testingLabelsDict = \
        loader.loadObjectFromPickle(pathToData)

    model = modeller.buildModel(WINDOW_LENGTH - PREDICTION_LENGTH, PREDICTION_LENGTH)

    epochCallback = LambdaCallback (on_epoch_end=lambda epoch, logs: saveModelToCloud(epoch, CHECKPOINT_PERIOD))

    model.fit(
        [
            trainingDataDict["weightedAverage"],
            trainingDataDict["volume"],
        ],
        [
            trainingLabelsDict["weightedAverage"]
        ],
        validation_data=(
        [
            testingDataDict["weightedAverage"],
            testingDataDict["volume"],
        ],
        [
            testingLabelsDict["weightedAverage"]
        ]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callback=[
        board.createTensorboardConfig(pathToJobDir + "/logs"),
        epochCallback
    ])
server.saveModelToCloud(model, pathToJobDir)
