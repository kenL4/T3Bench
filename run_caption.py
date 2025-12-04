from PIL import Image
import argparse
import os
import glob
from copy import deepcopy
import time
import openai
import cv2
import trimesh

from lavis.models import load_model_and_preprocess


openai.api_key = "PLEASE USE YOUR OWN API KEY"
MODEL = "gpt-4o-mini"

def predict(prompt, temperature, max_retries=10):
    for i in range(max_retries):
        try:

            response = openai.chat.completions.create(model=MODEL,
                                                        temperature=temperature,
                                                        messages=[
                                                            {"role": "user", "content": prompt}
                                                        ],)
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** i
            print(e)
            print("Retrying in", wait, "seconds")
            time.sleep(wait)

    raise RuntimeError("Could not distill the captions.")


def run(args, temp_path):

    device = 'cuda'
    model, vis_processors, _ = load_model_and_preprocess(name='blip2_t5', model_type='pretrain_flant5xxl', is_eval=True, device=device)

    Radius = 2.2
    icosphere = trimesh.creation.icosphere(subdivisions=0)
    icosphere.vertices *= Radius

    with open(f'data/prompt_{args.group}.txt') as f:
        lines = f.readlines()

    os.makedirs(f'outputs_caption', exist_ok=True)
    
    for prompt in lines:

        prompt = prompt.strip()

        try:
            obj_path = glob.glob(f'outputs_video/{args.method}_{args.group}/{prompt.replace(" ", "_")}/eval.mp4')[-1].replace("\'", "\\\'")
        except IndexError:
            obj_path = 'outputs_video/FALSE_PATH'

        capture = cv2.VideoCapture(obj_path)

        frame_id = 0
        captions = []
        while True:
            ret, frame = capture.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:, :512]
            color = Image.fromarray(frame_rgb).convert("RGB")
            image = vis_processors["eval"](color).unsqueeze(0).to(device)
            x = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=1)
            captions += x
            frame_id += 1
        capture.release()
    
        prompt_input = 'Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows:\n\n'
        for idx, txt in enumerate(captions):
            prompt_input += f'view{idx+1}: '
            prompt_input += txt
            prompt_input += '\n'
        prompt_input += '\nAvoid describing background, surface, and posture. The caption should be:'
        res = predict(prompt_input, 0)
        print(res)

        with open(f'outputs_caption/{args.method}_{args.group}.txt', 'a+') as f:
            f.write(prompt + ':' + res + '\n')
        
        

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
