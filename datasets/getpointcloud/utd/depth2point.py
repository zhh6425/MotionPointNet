import os
import numpy as np
import argparse
from matplotlib.image import imread
from glob import glob
import multiprocessing
import scipy.io as sio


'''
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

mkdir(args.output)
'''
#print('Action %02d finished!'%args.action)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth to Point Cloud')
    parser.add_argument('--input', default='DATA/video/UTD-MHAD/depth', type=str)
    parser.add_argument('--output', default='DATA/video/UTD-MHAD/video', type=str)
    # parser.add_argument('-n', '--action', type=int)
    args = parser.parse_args()

    W = 320
    H = 240

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    focal = 280

    files = 'DATA/video/UTD-MHAD/utd.list'
    with open(files, 'w') as f:
        for video_path in sorted(os.listdir(args.input)):
            video_name = video_path.replace('mat', 'npz')

            point_clouds = []
            video = sio.loadmat(os.path.join(args.input, video_path))['d_depth']

            nframes = str(video.shape[2])
            f.writelines(video_name + ' ' + nframes + '\n')

            for idx in range(int(nframes)):
                img = video[:, :, idx]  # (H, W)

                depth_min = img[img > 0].min()
                depth_map = img

                x = xx[depth_map > 0]
                y = yy[depth_map > 0]
                z = depth_map[depth_map > 0]
                x = (x - W / 2) / focal * z
                y = (y - H / 2) / focal * z

                points = np.stack([x, y, z], axis=-1)
                point_clouds.append(points)

            np.savez_compressed(os.path.join(args.output, video_name), data=point_clouds)

