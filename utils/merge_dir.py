import argparse
import os, random

if __name__ == "__main__":
    '''
    sample usage:
    python data_organizer.py \
        --src class1 class2 class3 \
        --dst /data/<some-target-name> \
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    src = args.src
    for lst in os.listdir(src):
        if not os.path.isdir(os.path.join(src, lst)):
            continue
        for file in os.listdir(os.path.join(src, lst)):
            if not file.endswith(".jpg"):
                continue
            new_file_name = lst + "_" + file
            print(new_file_name)
            os.system("sudo cp {src_img} {tgt_img}".format(
                src_img=os.path.join(src, lst, file),
                tgt_img=os.path.join(args.dst, new_file_name)
            ))
            