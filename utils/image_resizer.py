from PIL import Image
import argparse
import os

if __name__ == "__main__":
    '''
    sample usage:
    python data_organizer.py \
        --src /data/stl10/splitted-stl/train-before/ \
        --dst /home/ys3031/temp_data/train-before-resized \
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--dst", type=str)
    parser.add_argument("--new_width", type=int, default=640)
    parser.add_argument("--new_height", type=int, default=480)
    args = parser.parse_args()
    
    for label in os.listdir(args.src):
        
        src_label_dir = os.path.join(args.src, label)
        if not os.path.isdir(src_label_dir):
            continue
        
        dst_label_dir = os.path.join(args.dst, label)
        if not os.path.exists(dst_label_dir):
            os.makedirs(dst_label_dir)
        
        for img in os.listdir(src_label_dir):
            src_img_path = os.path.join(src_label_dir, img)
            src_img = Image.open(src_img_path)
            resized_img = src_img.resize((args.new_width, args.new_height), src_img.ANTIALIAS)
            resized_img.save(os.path.join(dst_label_dir, img))
            
    
    
    
    
    

    