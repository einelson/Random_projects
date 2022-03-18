'''
Example on opening and playing an audiofile via pyaudio and wav
'''
import pyaudio
import wave
import os

chunk = 1024

path = os.path.join(os.getcwd() + '/audio/rain.wav')

wf = wave.open(path, 'rb')

p=pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
data = wf.readframes(chunk)
print(data)

# Play the sound by writing the audio data to the stream
while data != '':
    stream.write(data)
    data = wf.readframes(chunk)

# Close and terminate the stream
stream.close()
p.terminate()