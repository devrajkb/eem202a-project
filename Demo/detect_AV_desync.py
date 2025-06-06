#https://medium.com/@umdfirecoml/a-step-by-step-guide-on-how-to-download-your-google-drive-data-to-your-jupyter-notebook-using-the-52f4ce63c66c

import argparse
import textwrap
from apiclient import discovery
from httplib2 import Http
from oauth2client import file, client, tools
import io
from googleapiclient.http import MediaIoBaseDownload
import subprocess
from scipy import signal
from tqdm import tqdm
import contextlib
import sys
import wave
import os
import webrtcvad
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

# grab the indexes of the facial landmarks for the mouth
# Note: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
(mStart, mEnd) = (49, 68)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# Note: http://dlib.net/train_shape_predictor.py.html
print("[INFO] loading facial landmark predictor...")


detector = dlib.get_frontal_face_detector()
predictor_file = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)

#10
FRAME_DURATION_MS = 10 

def visualize(path: str):

    # reading the audio file
    raw = wave.open(path)

    # reads all the frames
    # -1 indicates all or max frames
    signal_data = raw.readframes(-1)
    signal_data = np.frombuffer(signal_data, dtype ="int16")

    # gets the frame rate
    f_rate = raw.getframerate()

    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time_vals = np.linspace(
        0, # start
        len(signal_data) / f_rate,
        num = len(signal_data)
    )

    # using matplotlib to plot
    # creates a new figure
    plt.figure(1)

    # title of the plot
    plt.title("Sound Wave")

    # label of x-axis
    plt.xlabel("Time")

    # actual plotting
    plt.plot(time_vals, signal_data)

    # shows the plot
    # in new window
    plt.show(block=False)

def get_mouth_aspect_ratio(mouth):
    # MAR Equations using three vertical distances
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[9])  # 51, 59
    B = dist.euclidean(mouth[3], mouth[8])  # 52, 58
    C = dist.euclidean(mouth[4], mouth[7])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)

    # return the mouth aspect ratio
    return mar

# define one constants, for mouth aspect ratio to indicate open mouth
# MOUTH_AR_THRESH = 0.55

def mouth_aspect_ratio(frame):
    #frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    # If there are no faces, return zero
    if not rects:
        return 0

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)

    # extract the mouth coordinates, then use the
    # coordinates to compute the mouth aspect ratio
    mouth = shape[mStart:mEnd]
    return get_mouth_aspect_ratio(mouth)

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_classification(sample_rate, frame_duration_ms, vad, a_frames):
    is_speech = 0
    output_binary_list = []

    for frame in a_frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if (is_speech):
            output_binary_list.append(1)
        else:
            output_binary_list.append(0)
    return output_binary_list

# UNUSED
# def mouth_aspect_ratio_binary(frame, MOUTH_AR_THRESH):
#     mar = mouth_aspect_ratio(frame)
#     if mar > MOUTH_AR_THRESH:
#         # Mouth is open
#         return 1
#     # Mouth is closed
#     return 0

def get_voice_activity(audio, sample_rate, VAD_agg):
    vad = webrtcvad.Vad(VAD_agg)
    audio_frames = frame_generator(FRAME_DURATION_MS, audio, sample_rate)
    audio_frames = list(audio_frames)
    return vad_classification(sample_rate, FRAME_DURATION_MS, vad, audio_frames)

def extract_video(video_file):
    frames = []
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f'The frame rate is {frame_rate} FPS')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    return frames, frame_rate

def preprocess_video(frames, num_window_frames):
    # Specify feature extraction method
    get_video_features = mouth_aspect_ratio

    output_list = []
    for frame in tqdm(frames):
        # Get video features
        features = get_video_features(frame)
        output_list.append(features)

    # Smooth output
    b, a = signal.bessel(4, 0.2)
    smoothed = signal.filtfilt(b, a, output_list)

    # Normalize output  
    return (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed))

def to_binary(array, threshold):
    # Replaces elements >= threshold with 1 and the rest with 0
    return np.where(array >= threshold, 1, 0)

def extract_audio(video_file, audio_file):
    # Pull audio stream from video
    subprocess.call(['ffmpeg', '-y', '-i', video_file, '-ac', '1', '-ar', '16000', audio_file])
    return read_wave(audio_file)

def preprocess_audio(audio, sample_rate, num_video_frames, VAD_agg):
    # Get audio features										
    output_list = get_voice_activity(audio, sample_rate, VAD_agg)

    # Resize output to match length of frame list					 
    resampled = signal.resample(output_list, num_video_frames)

    # Smooth output		   
    b, a = signal.bessel(4, 0.2)
    smoothed = signal.filtfilt(b, a, resampled)

    # Get rid of overshoots in resampled signal
    output_binary = to_binary(smoothed, 0.5)

    return np.array(output_binary)

def main(argv):
    parser = argparse.ArgumentParser(
        description="Lip-sync detection via Mouth Aspect Ratio (MAR) and Voice Activity Detection (VAD).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example usage:
              python your_script.py path/to/video.mp4
              python your_script.py path/to/video.mp4 --vad_agg 2 --mar_thresh 0.45

            VAD Aggressiveness levels:
              0 - Very permissive (detects even soft speech)
              3 - Very aggressive (filters out most non-speech)
        """)
    )
    parser.add_argument("video_file", type=str, help="Path to the input video file (.mp4 or .avi)")
    parser.add_argument("--vad_agg", type=int, default=3, choices=[0, 1, 2, 3], help="VAD aggressiveness level (0-3), default=3")
    parser.add_argument("--mar_thresh", type=float, default=0.5, help="Mouth Aspect Ratio threshold for open mouth, default=0.5")

    if len(argv) == 0:
        parser.print_help()
        exit(1)

    args = parser.parse_args(argv)

    video_file = args.video_file
    VAD_agg = args.vad_agg
    MOUTH_AR_THRESHOLD = args.mar_thresh
    num_window_frames = 30

    print(f"[INFO] Video File: {video_file}")
    print(f"[INFO] VAD Aggressiveness: {VAD_agg}")
    print(f"[INFO] MAR Threshold: {MOUTH_AR_THRESHOLD}")

    start_time = int(round(time.time() * 1000))

    #Define file names for processing 
    video_dir = os.path.dirname(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    audio_file = os.path.join(video_dir, video_name + '.wav')
    output_video = os.path.join(video_dir, video_name + '_synced.mp4')

    #Extract video and audio
    frames, frame_rate = extract_video(video_file)
    audio, sample_rate = extract_audio(video_file, audio_file)

    #Preprocessing/feature extraction phase 
    Xv = preprocess_video(frames, num_window_frames)
    Xa = preprocess_audio(audio, sample_rate, len(frames), VAD_agg)

    # using matplotlib to plot
    plt.figure(6)
    plt.title("Xv")
    plt.xlabel("samples")
    plt.plot(Xv)
    plt.show(block=False)  

    plt.figure(7)
    plt.title("Xa")
    plt.xlabel("samples")
    plt.plot(Xa)
    plt.show(block=False)  

    #Correlation to find the time offset
    Xv_np = np.hstack(Xv)
    Xa_np = np.hstack(Xa)
    corr = np.correlate(Xv_np, Xa_np, mode='same')
    index = np.argmax(corr)
    computed_delay_frames = index - len(Xv_np) // 2
    computed_delay = 0.040 * computed_delay_frames

    plt.figure(8)
    plt.title("CORR")
    plt.xlabel("samples")
    plt.plot(corr)
    plt.show(block=False)  


    print(f"[INFO] Number of frames in video: {len(Xv_np)}")
    print(f"[INFO] Computed delays : {computed_delay_frames} frames")
    print(f"[INFO] Computed delays: {computed_delay:.3f} seconds")

    subprocess.call([
        'ffmpeg', '-y', '-i', video_file,
        '-itsoffset', str(computed_delay), '-i', video_file,
        '-map', '0:v', '-map', '1:a', '-c', 'copy', output_video
    ])

    elapsed = time.time() - (start_time / 1000)
    print(f"[INFO] Saved synced video to: {output_video}")
    print(f"[INFO] Processing took {elapsed:.2f} seconds")

    visualize(audio_file)

if __name__ == '__main__':
    main(sys.argv[1:])
    input("Press Enter to close graphs and end")

