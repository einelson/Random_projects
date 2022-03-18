'''
Convery an audiofile to numpy
https://www.geeksforgeeks.org/plotting-various-sounds-on-graphs-using-python-and-matplotlib/
'''
import os, wave
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# open file
path = os.path.join(os.getcwd() + '/audio/rain.wav')
wf = wave.open(path, 'rb')

# reads all the frames
# -1 indicates all or max frames
signal = wf.readframes(-1)
signal = np.frombuffer(signal, dtype ="int16")
    
# gets the frame rate
f_rate = wf.getframerate()
# to Plot the x-axis in seconds
# you need get the frame rate
# and divide by size of your signal
# to create a Time Vector
# spaced linearly with the size
# of the audio file
time = np.linspace(
    0, # start
    len(signal) / f_rate,
    num = len(signal)
)

# will plot signal by time
print(time)
print(signal)

plt.figure(1)
     
# title of the plot
plt.title("Sound Wave")
    
# label of x-axis
plt.xlabel("Time")

# actual plotting
plt.plot(time, signal)
    
# shows the plot
# in new window
plt.show()

# fig = px.line( x = time ,
#               y = signal,
#               title = 'A signal')
# fig.show()