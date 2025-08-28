import argparse
import logging
from datasets import load_dataset
from pathlib import Path
import requests
import PIL
import multiprocessing as mp


def download_sample(sample, dataset_prefix, log_file):
    # setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))

    # get sample data
    image_url = sample["image_url"]
    page_description = sample["context_page_description"]

    # Create a directory for the image
    image_base_path = Path(dataset_prefix) / str(hash(image_url))
    image_base_path.mkdir(parents=True, exist_ok=True)

    # prepare files paths
    image_path = image_base_path / "sample.img.jpg"
    page_description_path = image_base_path / "sample.page_description.txt"

    # check if image already exists
    if image_path.exists():
        return

    try:
        # Download the image if it doesn't exist
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        }

        response = requests.get(sample["image_url"], stream=True, headers=headers)
        if response.status_code == 200:
            # save image
            image = PIL.Image.open(response.content).convert("RGB")
            image.save(image_path, format="JPEG")

            # save page description
            page_description_path = image_base_path / "page_description.txt"
            page_description_path.write_text(page_description)

        else:
            logger.error(f"Failed to download image {sample['image_url']}")

    except Exception as e:
        logger.error(f"Failed to download or convert image {sample['image_url']}")


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_prefix", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--log_file", type=str, default="download_dataset.log")

    args = parser.parse_args()

    # load dataset
    dataset = load_dataset(
        "google/wit",
        split="train",
        cache_dir="./cache",
        trust_remote_code=True,
    )

    # download dataset
    with mp.Pool(processes=args.num_workers) as pool:
        pool.map(
            download_sample,
            zip(
                dataset,
                [args.dataset_prefix] * len(dataset),
                [args.log_file] * len(dataset),
            ),
        )
