import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from matplotlib import colors,cm
import json
import scipy.fftpack

def heartbeat(fps, sample, span):
    step = np.ceil(int(fps)//int(sample)) #number of frames between each frame given in data[]
    span_frames = round(span*fps/step) #number of data points required to cover a span.
    print(span_frames)
    Tracker = True
    Counter = 1
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load("facial_average.npy")
    np.load = np_load_old

    
    #Getting rid of a non-null dataset. The estimation of middle values could probably be improved using linear regression but this will help to smooth any changes out a bit.
    for x in range(len(data)):
        if data[x][0] == 0 and data[x][1]==0 and data[x][2]==0:
            if x > 0 and x <len(data) -1 and data[x+1][0] != 0 and data[x+1][1]!=0 and data[x+1][2]!=0:
                data[x] = (data[x-1] + data[x+1])/2
            elif x == 0:
                while tracker:
                    data[x] = data[x+Counter]
                    if data[x][0] != 0 and data[x][1]!=0 and data[x][2]!=0:
                        tracker = False
                        Counter = 0
                    counter += 1
            elif x == len(data) - 1:
                data[x] = data[x-1]
            elif data[x+1][0] == 0 and data[x+1][1]==0 and data[x+1][2]==00:
                data[x] = data[x-1]
    #Actual fourier transform portion of code. We will normalise over the span given with a step of 1 second to approximate a heart rate for that period of time. 
    #The steps are: 1. Normalise span 2. Use ICA to find source signals 3. Apply fourier transform to source signals to find appropriate frequencies.
    tempData = []
    ica = FastICA(n_components=3, max_iter=1000)
    time = np.linspace(0,span,span_frames)
    print(int(fps/(np.ceil(int(fps)//int(sample)))))
    count = 0
    data = data[:450]
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    x1 = [row[0] for row in data]
    #x1 = np.column_stack([x1,time.T[:34]])
    x2 = [row[1] for row in data]
    #x2 = np.column_stack([x2,time.T[:34]])
    x3 = [row[2] for row in data]
    #x3 = np.column_stack([x3,time.T[:34]])
    #x3.concatenate(time.T)
    source = np.c_[x1,x2,x3]
    S_ = ica.fit_transform(source)
    P_ = ica.inverse_transform(S_)

    for i in range(3):
        powerSpec = np.abs(np.fft.fft(S_[:,i]))
        #print(powerSpec)
        freqs = np.fft.fftfreq(len(S_[:,i]),1.0/fps)
        #print(freqs)
        validFreqs = np.where((freqs >= 0.75) & (freqs<= 4))
        validPower = powerSpec[validFreqs]
        #print(validPower)
        maxPower = np.argmax(validPower)
        print(freqs[validFreqs[0][maxPower]])
        #print('Max Power', maxPower)
        hr = freqs[maxPower]
        #print(hr)
    plt.plot(freqs[validFreqs],validPower)



    yf = scipy.fftpack.fft(S_[:,0])
    xf = np.linspace(0.0, 25.0/2.0, len(data)/2)
    #plt.plot(xf,2.0/len(data) * np.abs(yf[:len(data)//2]))


    fig = plt.figure()
    models = [source, S_, P_]
    names = ['data', 'source signal', 'predicted signal','frequencies']
    colors = ['blue', 'green','red']
    for i, (name, model) in enumerate(zip(names, models)):
        plt.subplot(4, 1, i+1)
        plt.title(name)
        for sig, color in zip (model.T, colors):
            plt.plot(sig, color=color, linewidth=0.5)

    
    fig.tight_layout()        
    plt.show()

if __name__ == "__main__":
    heartbeat(15,15,10)