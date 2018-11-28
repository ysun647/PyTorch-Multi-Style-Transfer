import argparse
import os, random

s = set()
flag = False
for i in range(300):
    for i in random.sample(range(5000), 10):
        if i in s:
            print(len(s))
            flag = True
            break
        s.add(i)
    if flag:
        break
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_1")
    parser.add_argument("--src_2")
    parser.add_argument("--dst")
    parser.add_argument("--num_train", type=int)
    parser.add_argument("--num_test", type=int)
    parser.add_argument("--class_1")
    parser.add_argument("--class_2")
    args = parser.parse_args()
    src_dir = args.src
    dst_dir = args.dst
    num = args.num

    for img in os.listdir(src_dir):
        if img.endswith(".jpg"):
            src_img = os.path.join(src_dir, img)
            dst_img = os.path.join(dst_dir, img)
            os.system("sudo cp %s %s" % (src_img, dst_img))
            num -= 1
            if num <= 0:
                break
#