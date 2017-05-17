import argparse
import subprocess
import os.path

# Executable dependencies: ffmpeg, ffprobe

def extract_audio_samples(filename, sample_length=10, out_dir=None):
    """
    Given a filename, extracts all 10-second samples as wav files
    """
    lengthCmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'
    sampleCmd = 'ffmpeg -ss {} -t {} -i "{}" {}'
    audioLength = float(subprocess.check_output(lengthCmd.format(filename), shell=True))
    fDir, fName = os.path.split(filename)
    fBase = os.path.splitext(fName)[0]
    for i in range(int(audioLength/sample_length)):
        outfile = os.path.join(out_dir or fDir, "{}.{}.wav".format(fBase, i))
        subprocess.call('ffmpeg -ss {} -t {} -i "{}" {}'.format(i*sample_length, sample_length, filename, outfile), shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="extract_audio_samples")
    parser.add_argument('filename')
    parser.add_argument('--out_dir')
    parser.add_argument('--sample_length', type=int, default=10)
    args = parser.parse_args()
    extract_audio_samples(args.filename, args.sample_length, out_dir=args.out_dir)
