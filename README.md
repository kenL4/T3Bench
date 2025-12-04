# T<sup>3</sup>Bench for Enhancer Evaluation

This is a modified version of T<sup>3</sup>Bench modified to evaluate enhanced GSGEN pipelines.

### I would highly recommend that you use the provided `T3Bench.jpynb` as the process is fairly complicated and has a lot of strict package requirements

If you want to run the benchmark locally, you should follow these steps:

#### Environment setup
The following environment has been found to work well with this program:
- Python 3.10 (Required)
- Torch 2.3.0
- CUDA 12.1
- C++17 support

#### Install dependencies

**[IMPORTANT]** There is a known [issue](https://github.com/THU-LYJ-Lab/T3Bench/issues/6), where salesforce-lavis and imagereward have directly conflicting package requirements so they are incompatible. The workaround in my fork is to remove salesforce-lavis from the requirements and then install it directly afterwards with `--no-deps` enabled. However, this does mean that we will need to install other packages depending on what script you are running.

Please run the following:
```
pip install -r T3Bench/requirements.txt
pip install tokenizers --no-cache-dir
pip install fairscale==0.4.4 timm==0.4.12 salesforce-lavis --no-deps
pip install "transformers<4.56" sacremoses==0.1.0 "tokenizers<0.22" -U --no-deps --no-cache-dir
```

### How to evaluate a model
For a given run of a model, save the video file from "gsgen/checkpoints/{prompt}/{date}/{uid}/eval/video" or save the GIF from Wandb and convert it to an MP4.
1. First, you should replace the prompts in T3Bench/data/prompt_single.txt with the prompts you want to evaluate.
2. Move the desired video file(s) to "T3Bench/outputs_mesh_t3/gsgen_single/{prompt}/eval.mp4"

#### Evaluate Quality
Run the following command
```
pip install "numpy<2" "huggingface_hub==0.25.2" diffusers transformers opencv-python clip -U

cd T3Bench/ && \
python run_eval_quality.py --group single --gpu 0 --method gsgen
```

#### Evaluate Alignment

**[IMPORTANT]**

From here on out, you will need an OpenAI API key, so **please replace the API key in run_caption.py and run_eval_alignment.py**

Run the following command
```
pip install decord openai einops --upgrade
pip install "transformers==4.30.2" "tokenizers<0.14" -U --no-deps --no-cache-dir

rm -rf outputs_caption
cd T3Bench/ && \
python3.10 run_caption.py --group single --gpu 0 --method gsgen
cd T3Bench/ && \
python run_eval_alignment.py --group single --gpu 0 --method gsgen
```

If you encounter an issue with VRAM while you do this, you can replace the BLIP2 model type in `run_caption.py`



### Citation

Please refer to the original authors instead of my fork

```
@misc{he2023t3bench,
      title={T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation}, 
      author={Yuze He and Yushi Bai and Matthieu Lin and Wang Zhao and Yubin Hu and Jenny Sheng and Ran Yi and Juanzi Li and Yong-Jin Liu},
      year={2023},
      eprint={2310.02977},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



### Acknowledgement

This project is a fork and therefore could not have been possible without the great open source work of T3Bench <a href="https://github.com/THU-LYJ-Lab/T3Bench">T3Bench</a>