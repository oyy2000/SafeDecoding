import os
import argparse
import pandas as pd
import json

# Define choices for multiple-choice questions
CHOICES = ["A", "B", "C", "D"]

# Mappings from subcategories to categories
SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

# Mappings from higher-level categories to their respective categories
CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other_business_health_misc": ["other", "business", "health"],
}

def format_subject(subject):
    """
    Formats the subject name by replacing underscores with spaces.

    Args:
        subject (str): The subject name with underscores.

    Returns:
        str: Formatted subject name.
    """
    return " ".join(subject.split("_")).strip()

def format_example(df, idx, include_answer=True):
    """
    Formats a single example from the dataframe into a prompt string and extracts the answer.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        idx (int): Index of the example in the DataFrame.
        include_answer (bool): Whether to include the answer in the prompt.

    Returns:
        tuple: A tuple containing the formatted prompt string and the answer.
    """
    prompt = df.iloc[idx, 0]
    num_choices = df.shape[1] - 2  # Assuming last column is the answer
    for j in range(num_choices):
        prompt += f"\n{CHOICES[j]}. {df.iloc[idx, j + 1]}"
    if include_answer:
        answer = df.iloc[idx, num_choices + 1].strip().upper()
        prompt += f"\nAnswer: {answer}\n\n"
    else:
        answer = None
        prompt += "\nAnswer:\n\n"
    return prompt, answer

def gen_prompt(train_df, subject, k=-1):
    """
    Generates a training prompt by concatenating formatted examples.

    Args:
        train_df (pd.DataFrame): DataFrame containing training examples.
        subject (str): The subject name.
        k (int, optional): Number of training examples to include. Defaults to -1 (all).

    Returns:
        str: Combined training prompt.
    """
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    else:
        k = min(k, train_df.shape[0])  # Ensure k does not exceed available examples
    for i in range(k):
        formatted_prompt, _ = format_example(train_df, i, include_answer=True)
        prompt += formatted_prompt
    return prompt

def get_subcategories_by_selected_categories(selected_categories):
    """
    Retrieves all subcategories that fall under the selected higher-level categories.

    Args:
        selected_categories (list): List of higher-level category names.

    Returns:
        set: Set of subcategory names that belong to the selected categories.
    """
    selected_subcategories = set()
    for high_cat in selected_categories:
        if high_cat not in CATEGORIES:
            print(f"Warning: High-level category '{high_cat}' not recognized. Skipping.")
            continue
        categories_in_high_cat = CATEGORIES[high_cat]
        for subcat, cats in SUBCATEGORIES.items():
            for cat in cats:
                if cat in categories_in_high_cat:
                    selected_subcategories.add(subcat)
    return selected_subcategories

def collect_attack_prompts(data_dir, selected_subcategories, ntrain=5):
    """
    Collects prompts and their corresponding answers from selected subcategories and aggregates them into a list,
    including category information.

    Args:
        data_dir (str): Path to the data directory containing 'dev' and 'test' folders.
        selected_subcategories (set): Set of subcategory names to include.
        ntrain (int): Number of training examples to include from the dev set.

    Returns:
        list: Aggregated list of dictionaries, each containing 'prompt', 'answer', 'subcategory', 'category', and optionally 'higher_level_category'.
    """
    attack_prompts = []
    
    for subcat in selected_subcategories:
        dev_path = os.path.join(data_dir, "dev", f"{subcat}_dev.csv")
        test_path = os.path.join(data_dir, "test", f"{subcat}_test.csv")
        
        # Check if both dev and test files exist
        if not os.path.isfile(dev_path):
            print(f"Development file for '{subcat}' not found at {dev_path}. Skipping this subcategory.")
            continue
        if not os.path.isfile(test_path):
            print(f"Test file for '{subcat}' not found at {test_path}. Skipping this subcategory.")
            continue
        
        # Load the CSV files
        dev_df = pd.read_csv(dev_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        
        # Limit the number of training examples if specified
        if ntrain > 0:
            train_examples = min(ntrain, dev_df.shape[0])
            dev_df = dev_df.iloc[:train_examples]
        else:
            train_examples = dev_df.shape[0]
        
        # Generate the training prompt
        train_prompt = gen_prompt(dev_df, subcat, k=ntrain)
        
        # Retrieve the immediate category for the subcategory
        immediate_categories = SUBCATEGORIES.get(subcat, ["other"])
        
        # Determine the higher-level categories
        higher_level_categories = []
        for high_cat, cats in CATEGORIES.items():
            if any(cat in immediate_categories for cat in cats):
                higher_level_categories.append(high_cat)
        
        # Collect prompts from the test set
        for i in range(test_df.shape[0]):
            # Generate the test prompt without the answer
            test_prompt, _ = format_example(test_df, i, include_answer=False)
            # Combine training prompts with the current test prompt
            full_prompt = train_prompt + test_prompt
            # Extract the answer from the test_df
            try:
                answer = test_df.iloc[i, -1].strip().upper()  # Assuming the last column is the answer
            except IndexError:
                print(f"Warning: Missing answer for subcategory '{subcat}', example index {i}. Skipping this example.")
                continue
            if answer in CHOICES:
                prompt_entry = {
                    "prompt": full_prompt,
                    "answer": answer,
                    "subcategory": subcat,
                    "category": immediate_categories
                }
                # Optionally include higher-level categories
                if higher_level_categories:
                    prompt_entry["higher_level_category"] = higher_level_categories
                
                attack_prompts.append(prompt_entry)
            else:
                print(f"Warning: Invalid answer '{answer}' for subcategory '{subcat}', example index {i}. Skipping this example.")
        
        print(f"Collected {test_df.shape[0]} prompts from subcategory '{subcat}'.")
    
    print(f"\nTotal prompts collected: {len(attack_prompts)}")
    return attack_prompts

def save_attack_prompts(attack_prompts, save_path):
    """
    Saves the aggregated attack prompts and answers to a JSON file.

    Args:
        attack_prompts (list): List of dictionaries with 'prompt' and 'answer' keys.
        save_path (str): Path to save the attack prompts JSON file.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(attack_prompts, f, ensure_ascii=False, indent=4)
    print(f"Attack prompts saved to {save_path}")

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Process MMLU data into attack_prompts list based on selected categories.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the 'dev' and 'test' subdirectories with CSV files."
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default="attack_prompts.json",
        help="File path to save the aggregated attack prompts in JSON format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--selected-subcategories",
        type=str,
        nargs='+',
        help="List of subcategory names to include (e.g., abstract_algebra mathematics)."
    )
    group.add_argument(
        "--selected-categories",
        type=str,
        nargs='+',
        help="List of higher-level category names to include (e.g., STEM humanities)."
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5,
        help="Number of training examples to include from each subcategory's dev set."
    )
    
    args = parser.parse_args()
    
    # Determine which subcategories to include
    if args.selected_categories:
        selected_categories = args.selected_categories
        selected_subcategories = get_subcategories_by_selected_categories(selected_categories)
        if not selected_subcategories:
            print("No valid subcategories found for the selected categories. Exiting.")
            return
    else:
        selected_subcategories = set(args.selected_subcategories)
    
    print(f"Selected subcategories: {selected_subcategories}")
    
    # Collect attack prompts
    attack_prompts = collect_attack_prompts(
        data_dir=args.data_dir,
        selected_subcategories=selected_subcategories,
        ntrain=args.ntrain
    )
    
    # Save the attack prompts to a JSON file
    save_attack_prompts(attack_prompts, args.save_file)

if __name__ == "__main__":
    main()