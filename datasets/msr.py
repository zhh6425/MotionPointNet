import os
import sys
import numpy as np
from torch.utils.data import Dataset

def clip_normalize(clip):
    pc = np.reshape(a=clip, newshape=[-1, 3])
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    clip = (clip - centroid) / m
    return clip

class MSRAction3D(Dataset):
    def __init__(self, root, cfg, split='train'):  # pretrain/train/test
        super(MSRAction3D, self).__init__()
        frames_per_clip = cfg['clip_len']
        step_between_clips = cfg['frame_step']
        num_points = cfg['num_points']

        self.videos = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if split == 'pretrain' and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:]) - 1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes - step_between_clips * (frames_per_clip - 1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            if split == 'train' and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            if split == 'test' and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                self.videos.append(video)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = split != 'test'
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        label = self.labels[index]

        clip = [video[t+i*self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = clip_normalize(np.array(clip))

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        # clip = clip / 300  # T, N, 3

        return clip.astype(np.float32), label, index
