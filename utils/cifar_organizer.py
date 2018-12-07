import os
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--src_dir")
    parser.add_argument("--info_file")
    parser.add_argument("--train_dir")
    parser.add_argument("--val_dir")
    args = parser.parse_args()
    with open(args.info_file, 'r') as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            num, label = line.strip().split(',')
            
            rand = random.random()
            if rand < args.train_ratio:
                tgt_dir = os.path.join(parser.train_dir, label)
            else:
                tgt_dir = os.path.join(parser.val_dir, label)
            
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            
            src_img = os.path.join(parser.src_dir, num+'.png')
            os.system("sudo cp {src_img} {tgt_dir}".format(src_img=src_img, tgt_dir=tgt_dir))
            
            if int(num) % 100 == 0:
                print("{} pics done!".format(num))
            