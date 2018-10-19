import numpy as np
import cv2
import pickle
import statistics
import sounddevice as sd
import soundfile as sf
import os

dark = 0.4
glow = 1.5

def editChannel(channel, audiomean):
    blur1 = cv2.GaussianBlur(channel, (3, 3), 0)
    median = np.median(blur1)
    sigma = np.std(blur1)
    mean = np.mean(blur1)

    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))

    edge1 = cv2.Canny(blur1, lower, upper)
    blur2 = cv2.GaussianBlur(edge1, (51,51), 0)
    blur2 = (blur2/(np.amax(blur2)+1) * 255) * audiomean
    finalmix = np.uint8(channel*0.4) + np.uint8(blur2*1.5)

    return finalmix

files = ['Mercy']
for f in files:
    print('Littifying ' + f )
    data, fs = sf.read(f+'mono.wav')
    cap = cv2.VideoCapture(f+'.mkv')
    out = cv2.VideoWriter(f+'Lit.mkv', cv2.VideoWriter_fourcc('a','v','c','1'), cap.get(cv2.CAP_PROP_FPS), (1920, 1080))
    videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video Frames: ' + str(videoLength))
    audio = []

    print('Extracting audio data...')
    i = 0
    while i < data.shape[0]:
        j = i + data.shape[0]/videoLength
        audio.append(np.abs(np.fft.rfft(data[round(i):round(j)], n = 48000)))
        i=j

    print('Video Time')
    i=0
    while(cap.isOpened()) and i < videoLength:
        print(i)
        ret, image = cap.read()
        redEdited = editChannel(image[:,:,2], (statistics.mean(audio[i][19:249]))/(79+(2*115)))
        greenEdited = editChannel(image[:,:,1], (statistics.mean(audio[i][250:749]))/(22+(2*29)))
        blueEdited = editChannel(image[:,:,0], (statistics.mean(audio[i][750:2500]))/(7+(2*10)))
        i += 1
        final = cv2.merge((blueEdited, greenEdited, redEdited))
        #cv2.imshow('final', final)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        out.write(np.uint8(final))

    cv2.destroyAllWindows()
    cap.release()
    out.release()

    #Old Code'''
'''
#out = cv2.VideoWriter('LitBoujeeTests.mkv', cv2.VideoWriter_fourcc('M','J','P', 'G'), 1, (1920, 1080))
audio = pickle.load(open('ft.p', 'rb'))
image = cv2.imread('Videos/Migos/Bad3133.jpg')
redEdited = editChannel(image[:, :, 2], (statistics.mean(audio[3133][19:249])) / (79 + (2 * 115)))
greenEdited = editChannel(image[:, :, 1], (statistics.mean(audio[3133][250:749])) / (22 + (2 * 29)))
blueEdited = editChannel(image[:, :, 0], (statistics.mean(audio[3133][750:2500])) / (7 + (2 * 10)))
final = cv2.merge((blueEdited, greenEdited, redEdited))
#out.write(data_u8)
cv2.imshow('frame', final)
cv2.waitKey()
cv2.destroyAllWindows()```

