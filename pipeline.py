import argparse
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import numpy as np
import os
import params
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from interpreteFrame import interpreteFrame


def parseVideo(iFile,oFile):
    """filename= /path/to/file
    Reutrns
    """
    clip = VideoFileClip(iFile)
    new_clip = clip.fl_image(lambda x: interpreteFrame(x))
    new_clip.write_videofile(oFile,audio=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process Video pipeline and print results.')
    parser.add_argument('--input', '-i', type=str,
                        help='Input video path',default='challenge_video.mp4')
    parser.add_argument('--output', '-o', type=str,
                        help='Output video path', default='output_videos/out.mp4')
    args = parser.parse_args()
    print(args.input)

    parseVideo(args.input,args.output)