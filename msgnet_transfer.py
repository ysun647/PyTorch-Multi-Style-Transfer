from msgnet_models import Net, Variable
from msgnet_util import tensor_load_rgbimage, preprocess_batch, tensor_save_bgrimage, id_generator
import torch
import argparse
import os, random

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
    for img in os.listdir(source_dir):
        if img.endswith(".jpg"):
            style_img = ""
            # print(os.listdir(source_dir))
            # print(os.listdir(style_dir))
            assert any(s.endswith(".jpg") for s in os.listdir(style_dir))
            while not style_img.endswith(".jpg"):
                style_img = random.choice(os.listdir(style_dir))
            target_img = os.path.join(target_dir, img[:-4] + "--" + style_img[:-4] + "--" + id_generator(4) + ".jpg")
            source_img = os.path.join(source_dir, img)
            style_img = os.path.join(style_dir, style_img)
            transfer_single_image(source_img,
                                  style_img,
                                  target_img,
                                  model_path=model_path)
            num -= 1
            
            if num % print_every == 0:
                print(print_every, " pics left")
            
            if num <= 0:
                break



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
          --src /Users/yiming/dev/data/pic/realism \
          --style <the-dir-that-stores-style-image> \
          --tgt /Users/yiming/dev/data/pic/realism-postimp-result \
          --model_path /Users/yiming/dev/data/models/coco_epoch_1.model \
          --num <how-many-pic-you-want-to-transfer>
        '''
        transfer_multi_image(source_dir=args.src,
                             style_dir=args.style,
                             target_dir=args.tgt,
                             num=args.num,
                             model_path=args.model_path,
                             print_every=args.print_every)
    else:
        raise ValueError("unknown mode: %s" % args.mode)
    