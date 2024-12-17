import subprocess
import sys

def run_script(script_path, *args):
    """Run a Python script with optional arguments."""
    try:
        subprocess.run(["python", script_path, *args], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
        sys.exit(1)

def main():
    # Step 1: Data loading and cleaning
    print("\nStep 1: Data loading and cleaning...")
    cleaning_done = input("Have you already completed the data cleaning? (yes/no): ").strip().lower()

    if cleaning_done in ["no", "n"]:
        print("\nRunning data loading and cleaning...")
        run_script("data_load+cleaning.py")
    elif cleaning_done in ["yes", "y"]:
        print("Skipping data loading and cleaning.")
    else:
        print("Invalid input. Please answer 'yes' or 'no'. Exiting.")
        sys.exit(1)

    # Step 2: Choose sentiment analysis method
    print("\nStep 2: Choose a sentiment analysis method:")
    print("1. RoBERTa (sentiment_roberta.py)")
    print("2. VADER (sentiment_vader.py)")
    print("3. BART-large (sentiment_bart_large_zero_shot_classification.py)")
    print("4. Fine-Tuned RoBERTa (sentiment_roberta_finetuned.py)")

    choice = input("Enter 1, 2, 3, or 4: ").strip()
    if choice == "1":
        print("\nRunning RoBERTa sentiment analysis...")
        sentiment_file = "data/RoBERTa_sentiment_wallstreetbets_data.csv"
        sentiment_analyzer = "RoBERTa"
        run_script("sentiment_RoBERTa.py")
    elif choice == "2":
        print("\nRunning VADER sentiment analysis...")
        sentiment_file = "data/vader_sentiment_wallstreetbets_data.csv"
        sentiment_analyzer = "VADER"
        run_script("sentiment_vader.py")
    elif choice == "3":
        print("\nRunning BART-large sentiment analysis...")
        sentiment_file = "data/bart_zero_shot_sentiment_wallstreetbets_data.csv"
        sentiment_analyzer = "BART"
        run_script("sentiment_bart_large_zero_shot_classification.py")
    elif choice == "4":
        print("\nRunning Fine-Tuned RoBERTa sentiment analysis...")
        sentiment_file = "data/roberta_finetuned_sentiment_wallstreetbets_data.csv"
        sentiment_analyzer = "Fine_Tuned_RoBERTa"
        run_script("sentiment_roberta_finetuned.py")
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Step 3: Stock alignment analysis
    print("\nStep 3: Running stock alignment analysis...")
    run_script("stock_allignment_multible_ex.py", sentiment_file, sentiment_analyzer)

    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main()
