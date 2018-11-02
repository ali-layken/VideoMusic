import numpy as np
import cv2
import statistics
import soundfile as sf
import argparse
import subprocess
import math

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('url', help='youtube url be proccessed')
parser.add_argument('output', help='output filename')
args = parser.parse_args()

def editChannel(channel, audiomean):
    blur1 = cv2.GaussianBlur(channel, (3, 3), 0)
    sigma = np.std(blur1)
    mean = np.mean(blur1)

    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))

    edge1 = cv2.Canny(blur1, lower, upper)
    blur2 = cv2.GaussianBlur(edge1, (51,51), 0)
    blur2 = (blur2/(np.amax(blur2)+1) * 255) * audiomean
    finalmix = np.uint8(channel*0.5) + np.uint8(blur2*1.5)

    return finalmix


def editChannel2(edgeog, channel, audiomean):
    mix = (edgeog/(np.amax(edgeog)+1) * 255) * audiomean
    finalmix  = np.uint8(channel*0.5) + np.uint8(mix*1.5)
    return finalmix


def auto_canny(blurredimg):
    median = np.median(blurredimg)
    sigma = np.std(blurredimg)
    mean = np.mean(blurredimg)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))
    edged = cv2.Canny(blurredimg, lower, upper)

    # return the edged image
    return edged

def edgeImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    auto = auto_canny(blurred)
    edgefinal = cv2.GaussianBlur(auto, (51, 51), 0)

    return edgefinal


def getFrameRate(filename):
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    rate = out.decode().split('=')[1].strip()[1:-1].split('/')
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1

def getFrameNum(filename):
    capNum = cv2.VideoCapture(filename)
    i = 0
    while (capNum.isOpened()):
        ret, img = capNum.read()
        if not ret:
            break
        i += 1
    capNum.release()
    return i

def downloadAndProcess(name, yurl):
    #Youtube donwload
    print('Downloading from youtube: \n')
    command = 'youtube-dl --merge-output-format mkv -x -k --audio-format wav -o "' + name + '.%(ext)s" ' + yurl
    subprocess.call(command, shell=True)

    command = 'ffmpeg -i ' + args.output + '.wav -ac 1 ' + args.output + 'mono.wav'
    subprocess.call(command, shell=True)

def main():
    print('Littifying ' + args.output)
    downloadAndProcess(args.output, args.url)

    #Opening Audio and Video files and grabbinng fps and other info
    print('Opening files and grabbing file info...')
    #audio
    data, fs = sf.read(args.output + 'mono.wav')
    #video
    cap = cv2.VideoCapture(args.output + '.mkv')

    #constants
    fps = getFrameRate(args.output + '.mkv')
    frameNum = getFrameNum(args.output + '.mkv')


    print('Processing video of ' + str(frameNum) +
          ' ' + str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) + 'x' + str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) +
          ' frames at ' + str(fps) + ' frames per second')

    #video output
    out = cv2.VideoWriter(args.output + 'Lit.mkv', cv2.VideoWriter_fourcc('a','v','c','1'), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    audio = []

    print('Extracting audio data...')
    i = 0

    while math.ceil(i) < data.shape[0]:
        j = i + data.shape[0]/frameNum
        audio.append(np.abs(np.fft.rfft(data[round(i):round(j)])))
        i=j

    #3 Break up into 3 audio ranges:
    b = 250
    m = 750
    t = 2500

    bass = [x for ff in audio for x in ff[0:b]]
    mid = [x for ff in audio for x in ff[b:m]]
    treb = [x for ff in audio for x in ff[m:t]]

    print('Doing Statistics...')
    bassMean = np.mean(bass)
    bassStdev = np.std(bass)
    print(bassMean, bassStdev)
    midMean = np.mean(mid)
    midStdev = np.std(mid)
    print(midMean, midStdev)
    trebMean = np.mean(treb)
    trebStdev = np.std(treb)


    print(trebMean, trebStdev)

    print('Video Processing time')

    i=0
    while(cap.isOpened()) and i < len(audio):
        print('Processing frame ' + str(i) + ' out of ' + str(frameNum))
        ret, image = cap.read()
        if not ret:
            break
        '''
        edgedimage = edgeImage(image)
        redEdited = editChannel2(edgedimage, image[:,:,2], (np.mean(audio[i][0:b]))/(bassMean+(2*bassStdev)))
        greenEdited = editChannel2(edgedimage, image[:,:,1], (np.mean(audio[i][b:m]))/(midMean+(2*midStdev)))
        blueEdited = editChannel2(edgedimage, image[:,:,0], (np.mean(audio[i][m:t]))/(trebMean+(2*trebStdev)))
        '''
        redEdited = editChannel(image[:,:,2], (np.mean(audio[i][0:b]))/(bassMean+(2*bassStdev)))
        greenEdited = editChannel(image[:,:,1], (np.mean(audio[i][b:m]))/(midMean+(2*midStdev)))
        blueEdited = editChannel(image[:,:,0], (np.mean(audio[i][m:t]))/(trebMean+(2*trebStdev)))

        i += 1
        final = cv2.merge((blueEdited, greenEdited, redEdited))
        out.write(np.uint8(final))

    cv2.destroyAllWindows()
    cap.release()
    out.release()
    print('\n\n\n')
    command = "ffmpeg -i " + args.output + "Lit.mkv -i " + args.output + ".wav -codec copy -shortest " + args.output + "LitFinal.mkv"
    subprocess.call(command, shell=True)
    '''

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
cv2.destroyAllWindows()'''


if __name__ == "__main__":
    main()
