import os
import glob
from datetime import datetime
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
from PIL import Image

def timer(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        func(*args, **kwargs)
        stop = datetime.now()
        print(f"The procedure {func.__name__} took {stop - start} to finish")
    return wrapper

def assert_same_name(A_path, B_path):
    A_name = os.path.basename(A_path).replace("_leftImg8bit", "")
    B_name = os.path.basename(B_path).replace("_gtFine_color", "")
    assert A_name == B_name, (f"Images {A_path} and {B_path} did not match ({A_name} != {B_name})")

def process_image(img_path, save_path):
    img = Image.open(img_path).convert("RGB").resize((512,512))
    img.save(save_path, format='JPEG', subsampling=0, quality=100)


def preprocess_func(paths):
    i, A_path, B_path, o_path, dataset = paths
    assert_same_name(A_path, B_path)
    process_image(A_path, os.path.join(o_path, "A", dataset, f"{i}_A.jpg"))
    process_image(B_path, os.path.join(o_path, "B", dataset, f"{i}_B.jpg"))

@timer
def preprocess(s_path, i_path, o_path, number_of_workers):
    datasets = ["train", "val"] #The test dataset will not be used

    for dataset in datasets:
        os.makedirs(os.path.join(o_path, "A", dataset), exist_ok=True)
        os.makedirs(os.path.join(o_path, "B", dataset), exist_ok=True)

        glob_A = os.path.join(i_path, dataset) + "/*/*_leftImg8bit.png"
        glob_B = os.path.join(s_path, dataset) + "/*/*_color.png"

        A_paths = sorted(glob.glob(glob_A))
        B_paths = sorted(glob.glob(glob_B))

        assert len(A_paths) == len(B_paths), (f"Dataset sizes does not match -> Orig: {len(A_paths)}, Sem: {len(B_paths)}")
        if number_of_workers > 1:
            with Pool(number_of_workers) as p:
                list(tqdm(p.imap(preprocess_func, zip(range(len(A_paths)), A_paths, B_paths, repeat(o_path), repeat(dataset))), total=len(A_paths), desc=f"[{dataset}] Preprocessed Pictures"))
        else:
            for i, (A_path, B_path) in tqdm(enumerate(zip(A_paths, B_paths)), total=len(A_paths), desc=f"[{dataset}] Preprocessed Pictures"):
                assert_same_name(A_path, B_path)
                process_image(A_path, os.path.join(o_path, "A", dataset, f"{i}_A.jpg"))
                process_image(B_path, os.path.join(o_path, "B", dataset, f"{i}_B.jpg"))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sem_seg_dir", "-s", type=str, required=True, help="Path to the semantic segmentation (Cityscape gtFine) directory")
    parser.add_argument("--orig_img_dir", "-i", type=str, required=True, help="Path to the original images (Cityscape gtFine) directory")
    parser.add_argument("--out_dir", "-o", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--num_of_workers", "-w", type=int, required=False, default=1, help="Number of workers to perform operation in parallel")

    options = parser.parse_args()

    print("Starting Preprocessing")
    preprocess(options.sem_seg_dir, options.orig_img_dir, options.out_dir, options.num_of_workers)
    print("Preprocessing Finished")