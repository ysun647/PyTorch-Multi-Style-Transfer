import argparse
import os, random


if __name__ == "__main__":
    '''
    sample usage:
    python data_organizer.py \
        --src class1 class2 class3 \
        --dst /data/<some-target-name> \
        --num 1000 \
        --train_ratio 0.2 \
        --train_name train \
        --val_name val
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs='+', type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--train_name", type=str, default="train")
    parser.add_argument("--val_name", type=str, default="val")
    args = parser.parse_args()
    
    assert (0 < args.train_ratio < 1)
    
    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(os.path.join(args.dst, args.train_name), exist_ok=False)
    os.makedirs(os.path.join(args.dst, args.val_name), exist_ok=False)
    
    for src in args.src:
        label_name = src.split("/")[-1]
        os.makedirs(os.path.join(args.dst, args.train_name, label_name))
        os.makedirs(os.path.join(args.dst, args.val_name, label_name))
        i = 0
        for file in os.listdir(src):
            if not file.endswith(".jpg"):
                continue
            rand = random.random()
            if rand < args.train_ratio:
                os.system("sudo cp {src_img} {tgt_img}".format(
                    src_img=os.path.join(src, file),
                    tgt_img=os.path.join(args.dst, args.train_name, label_name, file)
                ))
            else:
                os.system("sudo cp {src_img} {tgt_img}".format(
                    src_img=os.path.join(src, file),
                    tgt_img=os.path.join(args.dst, args.val_name, label_name, file)
                ))
            i += 1
            if i >= args.num:
                break

        