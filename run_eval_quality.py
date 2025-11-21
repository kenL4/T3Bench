import os
import glob
import argparse

import ImageReward as RM
import trimesh
import cv2
from PIL import Image


def run(args, temp_path):
    Radius = 2.2
    model = RM.load("ImageReward-v1.0")

    with open(f'data/prompt_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'result/quality', exist_ok=True)
    mean_score = 0
    
    for prompt in lines:
        prompt = prompt.strip()
        try:
            obj_path = glob.glob(f'outputs_video/{args.method}_{args.group}/{prompt.replace(" ", "_")}/eval.mp4')[-1].replace("\'", "\\\'")
        except IndexError:
            obj_path = 'outputs_video/FALSE_PATH'
        cap = cv2.VideoCapture(obj_path)

        frame_id = 0
        scores = {}
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:, :512]
            color = Image.fromarray(frame_rgb).convert("L")
            reward = model.score(prompt, color)
            # For some reason, T3Bench uses -114514 as the default
            scores[frame_id] = max(-114514, reward)
            frame_id += 1
        cap.release()

        # Unlike the original T3Bench, we now use frame neighbours
        radius = 1
        frame_ids = sorted(scores.keys())
        for _ in range(3):  # same 3-iteration smoothing
            new_scores = {}
            for i in frame_ids:
                neighbors = [scores[i]]

                # Convolve over frame neighbors within radius
                for r in range(1, radius + 1):
                    if i - r in scores:
                        neighbors.append(scores[i - r])
                    if i + r in scores:
                        neighbors.append(scores[i + r])
                new_scores[i] = sum(neighbors) / len(neighbors)
            scores = new_scores

        for idx in sorted(scores, key=lambda x: scores[x], reverse=True)[:1]:
            now_score = scores[idx] * 20 + 50
            mean_score += now_score / len(lines)
            print(now_score)

            with open(f'result/quality/{args.method}_{args.group}.txt', 'a+') as f:
                f.write(f'{now_score:.1f}\t\t{prompt}\n')

    print("Quality score:", mean_score)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='single', choices=['single', 'surr', 'multi'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, choices=['latentnerf', 'magic3d', 'fantasia3d', 'dreamfusion', 'sjc', 'prolificdreamer', 'gsgen'])
    args = parser.parse_args()

    i = 0
    while True:
        try: 
            temp_path = f'temp/temp_{i}'
            os.makedirs(temp_path)
            break
        except:
            i += 1
    
    run(args, temp_path)
    os.system(f'rm -r {temp_path}')
