import os
import argparse
import random

if __name__ == '__main__':
    '''
    sample usage:
    python sample_subdataset.py \
        --src /data/<source-name> \
        --dst /data/<some-target-name> \
        --sample_ratio 0.5 \
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--sample_ratio", type=float, default=0.5)
    parser.add_argument("--suffix", type=str, default='jpg')
    args = parser.parse_args()
    
    assert (0 < args.sample_ratio < 1)
    
    os.makedirs(args.dst, exist_ok=True)
    
    for img in os.listdir(args.src):
        if not img.endswith(args.suffix):
            print("found non {} file!".format(args.suffix))
            continue
            
        rand = random.random()
        if rand < args.sample_ratio:
            os.system("sudo cp {src_img} {tgt_img}".format(
                src_img=os.path.join(args.src, img),
                tgt_img=os.path.join(args.dst, img)
            ))
