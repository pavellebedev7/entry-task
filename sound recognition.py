from tkinter import *
from scipy.fftpack import fft
from scipy.signal import medfilt
from scipy import blackman
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

#  Constants
FS = 44100                  # Sample rate, s^-1
REC_TIME_0 = 1              # First record time, s
REC_TIME_1 = 5              # Second record time, s
LOGIC_LEVEL = 0.5
MATCH_LEVEL = 0.25
NOISE_LEVEL = 0.0002        # Spectrum noise filter set experimentally 0.0002
INPUT_NOISE_LEVEL = 0.01    # Input signal noise filter set = 1%
MEDFILT_W = 1               # Median filter window
N = 10                      # Number of chunks per second
input0 = 0
input1 = 0
output0 = 0
output1 = 0

#  Length calculation
LEN_0 = FS * REC_TIME_0     # First record length
LEN_1 = FS * REC_TIME_1     # Second record length
CHUNK = FS // N             # Chunk length
HALF_CHUNK = CHUNK // 2
DOUBLE_CHUNK = 2 * CHUNK
INPUT_TRIM = FS // 20       # Noisy start of a record set to 0
HALF_FS = FS // 2

#  Start and stop sounds
x = np.linspace(0, DOUBLE_CHUNK, DOUBLE_CHUNK)
start = 0.5 * np.sin(442.0 / FS * 2 * np.pi * x)
stop = 0.5 * np.sin(660.0 / FS * 2 * np.pi * x)


#  First record function
def rec0():
    global input0
    global LEN_0
    LEN_0 = FS * REC_TIME_0

    # Start sound
    sd.play(start, FS)
    sd.wait()

    # Record
    input0 = sd.rec(int(LEN_0), samplerate=FS, channels=1)
    sd.wait()
    input0[:INPUT_TRIM] = 0
    input0[np.abs(input0) <= INPUT_NOISE_LEVEL * np.amax(input0)] = 0
    input0 = np.trim_zeros(input0)

    # Stop sound
    sd.play(stop, FS)
    sd.wait()

    # Trim and filter
    if np.size(input0) < DOUBLE_CHUNK:
        input0 = np.append(input0, np.zeros(DOUBLE_CHUNK - np.size(input0)))
    else:
        input0 = np.append(input0, np.zeros((np.size(input0) // CHUNK + 1) * CHUNK - np.size(input0)))

    LEN_0 = np.size(input0)
    l5['text'] = LEN_0
    b1['text'] = "Record again"
    l3['text'] = 0
    play0()


#  First play function
def play0():
    sd.play(input0, FS)
    sd.wait()


#  Second record function
def rec1():
    global input1
    plt.close('all')

    # Start sound
    sd.play(start, FS)
    sd.wait()

    # Record
    input1 = sd.rec(int(LEN_1), samplerate=FS, channels=1)
    sd.wait()

    # Stop sound
    sd.play(stop, FS)
    sd.wait()

    # Trim and filter
    input1[:INPUT_TRIM] = 0
    input1[np.abs(input1) < INPUT_NOISE_LEVEL * np.amax(input1)] = 0
    b3['text'] = "Record again"
    play1()


#  Second play function
def play1():
    global output1
    global output0

    sd.play(input1, FS)
    sd.wait()

    x0 = np.arange(0, LEN_0, 1)
    x1 = np.arange(0, LEN_1, 1)
    y0 = np.reshape(input0, LEN_0)
    y1 = np.reshape(input1, LEN_1)

    w = blackman(CHUNK)

    #  Chunk numbers
    n0 = LEN_0 // CHUNK
    n1 = LEN_1 // CHUNK

    zf0 = np.zeros((n0, HALF_CHUNK))
    zf1 = np.zeros((n1, HALF_CHUNK))
    output0 = np.zeros((n0, HALF_CHUNK))
    output1 = np.zeros((n1, HALF_CHUNK))

    #  Spectrum calculation for first record
    for i in range(0, n0):
        zf0[i] = 2.0 / CHUNK * abs(fft(y0[CHUNK * i:CHUNK * (i + 1)] * w)[0:HALF_CHUNK])
        output0[i] = medfilt(zf0[i], MEDFILT_W)
        max0 = np.amax(output0[i])
        for j in range(0, HALF_CHUNK):
            #  Spectrum filter and mask
            if (output0[i][j] <= LOGIC_LEVEL * max0) or (np.abs(output0[i][j]) < NOISE_LEVEL):
                output0[i][j] = 0
            else:
                output0[i][j] = 1
    #  Spectrum calculation for second record
    for i in range(0, n1):
        zf1[i] = 2.0 / CHUNK * abs(fft(y1[CHUNK * i:CHUNK * (i + 1)] * w)[0:HALF_CHUNK])
        output1[i] = medfilt(zf1[i], MEDFILT_W)
        max1 = np.amax(output1[i])
        for j in range(0, HALF_CHUNK):
            #  Spectrum filter and mask
            if (output1[i][j] <= LOGIC_LEVEL * max1) or (np.abs(output1[i][j]) < NOISE_LEVEL):
                output1[i][j] = 0
            else:
                output1[i][j] = 1

    #  Spectrum comparison
    n = n1 + 1  # Number of comparisons
    xd = np.arange(n)
    matches = np.zeros(n)
    recognition = np.zeros(n)
    output1 = np.concatenate([output1, np.zeros((n0, HALF_CHUNK))])
    match_output1 = np.zeros((n0, HALF_CHUNK))
    for i in range(0, n):
        window = output1[i:i + n0, :]
        for j in range(0, n0):
            for k in range(0, HALF_CHUNK):
                match_output1[j][k] = window[j][k] and output0[j][k]
        total_harm = np.count_nonzero(output0)
        match_harm = np.count_nonzero(match_output1)
        # Match function
        if total_harm > 0:
            matches[i] = 1 - (total_harm - match_harm) / total_harm
        else:
            matches[i] = 0
        if i > 0:
            if ((matches[i] - matches[i - 1]) <= 0) and (matches[i - 1] > MATCH_LEVEL):
                recognition[i] = recognition[i - 1] = 1

    if np.count_nonzero(matches) == 0:
        l3['text'] = "No matches found"
    else:
        l3['text'] = np.count_nonzero(matches)

    #  Input/output plotting
    plt.figure()
    plt.title('Blue - first record, orange - second record, green - match')
    plt.plot(x0 / FS, y0, x1 / FS, y1)  # , xd * n1 / ((n - 1) * N), matches
    plt.grid(True)
    plt.fill_between(xd * n1 / ((n - 1) * N), -1, 1, where=recognition > 0, color='green', alpha='0.75')
    plt.xlabel('Time, s')
    plt.show()


#  Interface
root = Tk()
root.title("Sound recording and recognition")
root.geometry('1000x500')
l1 = Label(text="Record 1 = 1 s, Record 2 = 5 s\nRecognition only works for sounds that longer then 0.05 s", height=3)
l2 = Label(text="Matches found:", width=20, height=3)
l3 = Label(text="0")
l4 = Label(text="Recorded samples 1:", width=20, height=3)
l5 = Label(text="0")
b1 = Button(text="Record 1", width=20, height=3)
b2 = Button(text="Play 1", width=20, height=3)
b3 = Button(text="Record 2", width=20, height=3)
b4 = Button(text="Play 2", width=20, height=3)

b1.config(command=rec0)
b2.config(command=play0)
b3.config(command=rec1)
b4.config(command=play1)

l1.pack()
b1.pack()
b2.pack()
b3.pack()
b4.pack()
l2.pack()
l3.pack()
l4.pack()
l5.pack()

root.mainloop()
