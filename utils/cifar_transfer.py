from msgnet_models import Net, Variable
from msgnet_util import tensor_load_rgbimage, preprocess_batch, tensor_save_bgrimage, id_generator
import torch
import argparse
import os
import random
from datetime import datetime

def transfer_single_image(source_img, style_img, target_img, model_path='21styles.model'):
    content_image = tensor_load_rgbimage(source_img, size=512, keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage(style_img, size=512).unsqueeze(0)
    style = preprocess_batch(style)
    model_dict = torch.load(model_path)
    model_dict_clone = model_dict.copy()  # We can't mutate while iterating
    
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    
    style_model = Net(ngf=128)
    style_model.load_state_dict(model_dict, False)
    
    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    tensor_save_bgrimage(output.data[0], target_img, False)

def transfer_multi_image(source_dir, style_dir, target_dir, num, model_path, print_every=20):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    assert any(s.endswith(".jpg") for s in os.listdir(style_dir))
    for label in os.listdir(source_dir):
        label_dir = os.path.join(source_dir, label)
        if not os.path.isdir(label_dir):continue
        label_tgt_dir = os.path.join(target_dir, label)
        os.makedirs(label_tgt_dir)
        for img in os.listdir(label_dir):
            if not img.endswith(".png"): continue
            style_imgs = random.sample(os.listdir(style_dir), num)
            for style_img in style_imgs:
                target_img = os.path.join(label_tgt_dir, img[:-4] + "--" + style_img[:-4] + "--" + id_generator(4) + ".png")
                source_img = os.path.join(label_dir, img)
                style_img = os.path.join(style_dir, style_img)
                transfer_single_image(source_img,
                                      style_img,
                                      target_img,
                                      model_path=model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--src")
    parser.add_argument("--style")
    parser.add_argument("--tgt")
    parser.add_argument("--model_path")
    parser.add_argument("--num", type=int)
    parser.add_argument("--print_every", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "single":
        # sample usage: $ python msgnet_transfer.py --mode single --src candy.jpg --style venice-boat.jpg --tgt sym_output.jpg
        transfer_single_image(source_img=args.src,
                              style_img=args.style,
                              target_img=args.tgt,
                              model_path=args.model_path)
    elif args.mode == "multi":
        '''
        python msgnet_transfer.py \
          --mode multi \
          --src /data/cifar/train \
          --style /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/images/21styles \
          --tgt /data/cifar/augmented \
          --model_path /home/ys3031/PyTorch-Multi-Style-Transfer/experiments/models/cifar-1/Final_epoch_1_Fri_Dec__7_23:23:54_2018_1.0_5.0.model \
          --num 10
        '''
        transfer_multi_image(source_dir=args.src,
                             style_dir=args.style,
                             target_dir=args.tgt,
                             num=args.num,
                             model_path=args.model_path,
                             print_every=args.print_every)
    else:
        raise ValueError("unknown mode: %s" % args.mode)
