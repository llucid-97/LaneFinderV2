import argparse
import imageio

imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from interpreteFrame import interpreteFrame


def parse_video(i_file, o_file):
    """
    Runs the lane finding pipeline,
    iFile= /path/to/input_video.mp4
    oFile= /path/to/output_video.mp4
    """
    clip = VideoFileClip(i_file)
    new_clip = clip.fl_image(lambda x: interpreteFrame(x))
    new_clip.write_videofile(o_file, audio=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Video pipeline and print results.')
    parser.add_argument('--input', '-i', type=str,
                        help='Input video path', default='project_video.mp4')
    parser.add_argument('--output', '-o', type=str,
                        help='Output video path', default='output_videos/c_out.mp4')
    args = parser.parse_args()
    print(args.input)

    parse_video(args.input, args.output)
