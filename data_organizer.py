import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--dst")
    parser.add_argument("--num", type=int)
    args = parser.parse_args()
    src_dir = args.src
    dst_dir = args.dst
    num = args.num
    
    for img in os.listdir(src_dir):
        if img.endswith(".jpg"):
            src_img = os.path.join(src_dir, img)
            os.system("sudo cp %s %s" % (src_img, dst_dir))
            num -= 1
            if num <= 0:
                break
    