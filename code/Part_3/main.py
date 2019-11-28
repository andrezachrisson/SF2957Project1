import numpy as np
from keras.utils import to_categorical
from sklearn import datasets
from keras.layers import Input, Dense, Flatten, Activation, Conv2D, MaxPool2D
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
sns.set()

def plotSettings(fig1, fig2, fig3, fig4, ax1, ax2, ax3, ax4, type, folder):
    ax1.set(xlabel="Epochs", ylabel="Error", ylim=(-0.1,1))
    ax2.set(xlabel="Epochs", ylabel="Error", ylim=(-0.1,1))
    ax3.set(xlabel="Epochs", ylabel="Loss")
    ax4.set(xlabel="Epochs", ylabel="Loss")
    ax1.set_title("Error for training")
    ax2.set_title("Error for validation")
    ax3.set_title("Loss for training")
    ax4.set_title("Loss for validation")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    fig1.savefig(f"{folder}/{type}Acc.png")
    fig2.savefig(f"{folder}/{type}ValAcc.png")
    fig3.savefig(f"{folder}/{type}pLoss.png")
    fig4.savefig(f"{folder}/{type}ValLoss.png")


def dataCol(N, X, y):
    Ntrain = np.int(N)
    Ntest = np.int(100)
    Xtrain = X[0:Ntrain,:]
    ytrain = y[0:Ntrain]
    # print(X.shape[0])
    test = np.random.choice(np.arange(Ntrain, X.shape[0]), Ntest, replace=False)
    Xtest = X[test,:]
    ytest = y[test]

    yTrainCat = to_categorical(ytrain)
    yTestCat = to_categorical(ytest)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 8, 8, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], 8, 8, 1)

    return Xtest,Xtrain, yTrainCat, yTestCat


def main():
    digits = datasets.load_digits()
    X = digits.data
    y = np.array(digits.target, dtype = int)
    N,d = X.shape
    Xtest,Xtrain, yTrainCat, yTestCat = dataCol(N=N-100, X=X, y=y)

    # CNN model
    # Both these models where run with and without regularization dont forget to change folder name
    folderName="regularized"
    #folderName ="notRegularized"

    # 1 layer without pooling
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for idx, nrnodes in enumerate(range(10,40,5)):
        inputlayer = Input(shape = (8,8,1))
        convLayer1 = Conv2D(nrnodes,3, strides=(1,1), padding='same', activation='relu')(inputlayer)
        flattenLayer = Flatten()(convLayer1)
        outputLayer = Dense(10,activation='softmax')(flattenLayer)

        model = Model(inputlayer, outputLayer)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

        oneLayerPool = model.fit(x=Xtrain, y = yTrainCat, batch_size=10, epochs= 50,validation_data=(Xtest,yTestCat), verbose=0)
        # pdb.set_trace()
        ax1.plot(np.ones(len(oneLayerPool.history['acc']))-oneLayerPool.history['acc'], linestyle='--', label=f"{nrnodes} nodes, {np.round(1-oneLayerPool.history['acc'][-1],3)}")
        ax2.plot(np.ones(len(oneLayerPool.history['val_acc']))-oneLayerPool.history['val_acc'], linestyle='--', label=f"{nrnodes} nodes, {np.round(1-oneLayerPool.history['val_acc'][-1],3)} ")
        ax3.plot(oneLayerPool.history['loss'], linestyle='--', label=f"{nrnodes} nodes, {np.round(oneLayerPool.history['loss'][-1],3)}")
        ax4.plot(oneLayerPool.history['val_loss'], linestyle='--', label=f"{nrnodes} nodes, {np.round(oneLayerPool.history['val_loss'][-1],3)}")

    plotSettings(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, ax1=ax1, ax2=ax2, ax3=ax3, ax4=ax4, type ="Simp", folder=folderName)

    # # 2 convolutional layers 2 maxpool layers
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    for idx, nrnodes in enumerate(range(10,40,5)):
        inputlayer = Input(shape = (8,8,1))
        convLayer1 = Conv2D(nrnodes,3, strides=(1,1), padding='same', activation='relu', activity_regularizer=l2(0.001))(inputlayer)
        maxPool1 = MaxPool2D(pool_size=(2,2))(convLayer1)
        convLayer2 = Conv2D(int(nrnodes/2),3, strides=(1,1), padding='same', activation='relu', activity_regularizer=l2(0.001))(maxPool1)
        maxPool2 = MaxPool2D(pool_size=(2,2))(convLayer2)

        flattenLayer = Flatten()(maxPool2)
        outputLayer = Dense(10,activation='softmax')(flattenLayer)

        model = Model(inputlayer, outputLayer)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        oneLayerPool = model.fit(x=Xtrain, y = yTrainCat, batch_size=10, epochs= 50,validation_data=(Xtest,yTestCat), verbose=0)


        ax1.plot(np.ones(len(oneLayerPool.history['acc']))-oneLayerPool.history['acc'], linestyle='--', label=f"{nrnodes} nodes, {np.round(1-oneLayerPool.history['acc'][-1] ,3) }")
        ax2.plot(np.ones(len(oneLayerPool.history['val_acc']))-oneLayerPool.history['val_acc'], linestyle='--', label=f"{nrnodes} nodes, {np.round(1-oneLayerPool.history['val_acc'][-1],3)}")
        ax3.plot(oneLayerPool.history['loss'], linestyle='--', label=f"{nrnodes} nodes, {np.round(oneLayerPool.history['loss'][-1],3)}")
        ax4.plot(oneLayerPool.history['val_loss'], linestyle='--', label=f"{nrnodes} nodes, {np.round(oneLayerPool.history['val_loss'][-1],3)}")
    plotSettings(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, ax1=ax1, ax2=ax2, ax3=ax3, ax4=ax4, type ="Comp", folder =folderName)

    trainsize = [20,50, 100, 500, 1000, 1500]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    error= np.array([])
    errorVal = np.array([])
    errorComp = np.array([])
    errorValComp = np.array([])
    for N in trainsize:
        Xtest,Xtrain, yTrainCat, yTestCat = dataCol(N=N, X=X, y=y)

        # Simple model
        inputlayer = Input(shape = (8,8,1))
        convLayer1 = Conv2D(20,3, strides=(1,1), padding='same', activation='relu', activity_regularizer=l2(0.001))(inputlayer)
        flattenLayer = Flatten()(convLayer1)
        outputLayer = Dense(10,activation='softmax')(flattenLayer)

        model = Model(inputlayer, outputLayer)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        oneLayerPool = model.fit(x=Xtrain, y = yTrainCat, batch_size=10, epochs= 50,validation_data=(Xtest,yTestCat), verbose=0)
        error = np.append(error, 1-oneLayerPool.history['acc'][-1])
        errorVal = np.append(errorVal, 1-oneLayerPool.history['val_acc'][-1])
        plot_model(model, to_file='simpleModel.png', show_shapes=True, show_layer_names=True)
        print(model.count_params())

        # Complex model
        inputlayer = Input(shape = (8,8,1))
        convLayer1 = Conv2D(25,3, strides=(1,1), padding='same', activation='relu', activity_regularizer=l2(0.001))(inputlayer)
        maxPool1 = MaxPool2D(pool_size=(2,2))(convLayer1)
        convLayer2 = Conv2D(int(25/2),3, strides=(1,1), padding='same', activation='relu', activity_regularizer=l2(0.001))(maxPool1)
        maxPool2 = MaxPool2D(pool_size=(2,2))(convLayer2)

        flattenLayer = Flatten()(maxPool2)
        outputLayer = Dense(10,activation='softmax')(flattenLayer)

        model = Model(inputlayer, outputLayer)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        oneLayerPool = model.fit(x=Xtrain, y = yTrainCat, batch_size=10, epochs= 50,validation_data=(Xtest,yTestCat), verbose=0)
        errorComp = np.append(errorComp, 1-oneLayerPool.history['acc'][-1])
        errorValComp = np.append(errorValComp, 1-oneLayerPool.history['val_acc'][-1])
        plot_model(model, to_file='complexModel.png', show_shapes=True, show_layer_names=True)
        print(model.count_params())


    ax1.plot(trainsize,error,linestyle='--', marker='o', label=f"Simple Model")
    ax1.plot(trainsize,errorComp,linestyle='--', marker='o', label=f"Complex Model")
    ax2.plot(trainsize,errorVal,linestyle='--', marker='o', label=f"Simple Model")
    ax2.plot(trainsize,errorValComp,linestyle='--', marker='o', label=f"Complex Model")

    ax1.set(xlabel="Training size", ylabel="Error", ylim=(-0.1,1))
    ax1.set_title("Error for training")
    ax1.legend()
    ax2.set(xlabel="Training size", ylabel="Error", ylim =(-0.1,1))
    ax2.set_title("Error for validation")
    ax2.legend()
    fig1.savefig(f"dataSizeAcc.png")
    fig2.savefig(f"dataSizeValAcc.png")





if __name__== "__main__":
    main()
