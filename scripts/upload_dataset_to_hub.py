from huggingface_hub import HfApi
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="Local dataset folder path")
    parser.add_argument("repo_id", help="HF repo id, e.g. OliverHausdoerfer/my-dataset")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    repo_id = api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True).repo_id
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=args.folder_path,
        commit_message="Upload dataset",
    )
    print(f"Uploaded to {commit.commit_url}")


if __name__ == "__main__":
    main()
