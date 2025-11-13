import argparse
import os
import datasets


thinking_format_instruction = (
    "You are a safe and helpful multimodal assistant. "
    "Given the user's question and image, you should reason carefully before responding.\n"
    "Follow these reasoning steps:\n"
    "Step 1: Caption the image and understand what it depicts.\n"
    "Step 2: Analyze whether the visual content itself contains unsafe, disturbing, or risky elements. "
    "After reasoning this step, output <visual_safe>safe</visual_safe> or <visual_safe>unsafe</visual_safe>.\n"
    "Step 3: Analyze whether the textual instruction contains unsafe intent or encourages unsafe actions. "
    "After reasoning this step, output <text_safe>safe</text_safe> or <text_safe>unsafe</text_safe>.\n"
    "Step 4: Combine the visual and textual information to determine whether their interaction could produce or imply any unsafe, misleading, illegal, or harmful meaning, action, or outcome. "
    "After reasoning this step, output <combined_safe>safe</combined_safe> or <combined_safe>unsafe</combined_safe>.\n"
    "Step 5: Summarize your reasoning about the overall safety of the content and possible causes of risk.\n"
    "Then, based on your reasoning results:\n"
    "- If <combined_safe>unsafe</combined_safe>, politely refuse to answer the question and briefly explain why.\n"
    "- If <combined_safe>safe</combined_safe>, provide a normal, safe, and helpful response to the user's instruction.\n\n"
    "The reasoning process MUST be enclosed within <think> </think> tags. "
    "The final answer MUST be enclosed within <answer> </answer>."
)


def process_fn(example, idx):
    problem = example.pop("prompt")
    prompt = problem + "\n" + thinking_format_instruction
    images = example.pop("images")
    text_tag = example.pop("text_tag")
    image_tag = example.pop("image_tag")
    combine_tag = example.pop("combine_tag")

    return {
        "data_source": args.dataset_name,
        "prompt": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "images": images,
        "ability": "safety",
        "reward_model": {
            "style": "rule",
            "ground_truth": ""
        },
        "extra_info": {
            "index": idx,
            "text_tag": text_tag,
            "image_tag": image_tag,
            "combine_tag": combine_tag,
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, default="XuankunRong/SafeTag-VL-3K")
    parser.add_argument("--local_save_dir", type=str, required=True, default="./train_data")
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    for split in ["train", "test"]:
        print(f"🚀 Processing split: {split}")
        dataset = datasets.load_dataset(args.dataset_name, split=split)
        dataset = dataset.map(function=process_fn, with_indices=True, num_proc=8)

        parquet_path = os.path.join(args.local_save_dir, f"safetygrpo_{split}.parquet")
        dataset.to_parquet(parquet_path)
        print(f"✅ Saved parquet to {parquet_path}")