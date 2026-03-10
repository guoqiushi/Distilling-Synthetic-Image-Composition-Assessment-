# SynthCompQA: Synthetic-Composition-Quality-Assessment

## Introduction
In this project, we propose an approach for estimating the quality of synthetic images generated through image composition. Unlike traditional image composition assessment methods, which primarily focus on aesthetics and distortion, our approach emphasizes semantic consistency and spatial consistency in synthetic images. These factors are particularly important in data augmentation pipelines, where the usefulness of synthetic data depends not only on visual plausibility but also on semantic and spatial correctness.
## Data Preparation
### Data Source
- [WIDER](http://shuoyang1213.me/WIDERFACE/)
- [PASCAL VOC](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/)
- [Object365](https://www.objects365.org/overview.html)
### Synthetic image Generation
Mask Gallery: Using [sam3](sam3_text_prompt_to_transparent_png_mp.py) to extract transparent mask 
Random Copy-Paste to genate synthetic images
## Getting start
1. Split synthetic images into 4 groups: **perfect**,**good**,**medium** and **low**.
2. Generate train.jsonl and val.jsonl using code [here](gen_jsonl.ipynb)
3. Finetune Qwen3.5-2B
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --main_process_port 29501 --debug train_.py
```
4. Generate Soft labels: [scripts](export_soft_lable.py)
5. Train student net
```bash
python train_student.py 
```
