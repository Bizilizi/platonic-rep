from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import requests
from torch.utils.data import Dataset
from PIL import Image
import os
from datasets import load_dataset
import random
import json

# define model and processor
device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    cache_dir="/storage/slurm/zverev/models",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


# load dataset
class WITDataset(Dataset):
    def __init__(self, dataset, dataset_prefix):
        self.dataset = dataset
        self.dataset_prefix = dataset_prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_name = sample["image_url"].split("/")[-1]

        # Create a directory for the image
        image_path = Path(self.dataset_prefix) / str(hash(sample["image_url"]))
        image_path.mkdir(parents=True, exist_ok=True)
        image_path = image_path / image_name

        # Download the image if it doesn't exist
        if not image_path.exists():
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
            }

            response = requests.get(sample["image_url"], stream=True, headers=headers)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image {sample['image_url']}")
                image_path = ""

        sample["image_path"] = str(image_path)

        return sample


dataset = load_dataset(
    "google/wit",
    split="train",
    cache_dir="/storage/slurm/zverev/datasets",
    trust_remote_code=True,
)
wit_datasets = WITDataset(dataset, dataset_prefix="/storage/slurm/zverev/datasets/WIT")

# load indices
activations_prefix = "./activations-text+img"

if not os.path.exists("./indices.json"):
    indices = random.sample(range(len(wit_datasets)), 10_000)
    with open("./indices.json", "w") as f:
        json.dump(indices, f)
else:
    with open("./indices.json", "r") as f:
        indices = json.load(f)

# compute activations

for i in tqdm(indices):
    try: 
        sample = wit_datasets[i]
        
        # prepare path
        url_hash, image_name = sample["image_path"].split("/")[-2:]
        activations_path = Path(activations_prefix) / url_hash
        
        # skip already processed samples
        if activations_path.exists():
            continue
        
        activations_path.mkdir(parents=True)
    
        if sample["image_url"].endswith(".svg") or sample["image_path"] == "":
            continue

        # Process the conversation
        inputs = processor(
            text="Describe what is in this image: <image>",
            images=[Image.open(sample["image_path"])],
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt" 
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
        with torch.inference_mode():
            output_ids = model(**inputs, output_hidden_states=True)
        
        # save hidden states 
        hidden_states = torch.stack(
            [hs[0].cpu() for hs in output_ids.hidden_states]
        )
        
        image_hidden_states = hidden_states[:, 8:]
        torch.save(image_hidden_states.mean(dim=1), activations_path / "image_hidden_states_avg.pt")
        torch.save(image_hidden_states[:, -1].clone(), activations_path / "image_hidden_states_last.pt")
        
        text_hidden_states = hidden_states[:, :8]
        torch.save(text_hidden_states.mean(dim=1), activations_path / "text_hidden_states_avg.pt")
        torch.save(text_hidden_states[:, -1].clone(), activations_path / "text_hidden_states_last.pt")
        
    except KeyboardInterrupt:
        break 
    except Exception as e:
        print(f"Skip index {i}")