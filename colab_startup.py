import os
import subprocess

from google.colab import drive  # ty:ignore[unresolved-import]


def setup_github_environment(repo_name, user_name, user_email):
    """
    Mounts Google Drive, navigates to the repository,
    and configures Git credentials for the session.
    """
    # 1. Mount Google Drive
    print("Mounting Google Drive...")
    drive.mount("/content/drive")

    # 2. Define and verify the project path
    project_path = f"/content/drive/MyDrive/{repo_name}"

    if not os.path.exists(project_path):
        raise FileNotFoundError(
            f"The directory {project_path} does not exist. "
            f"Please ensure the repository was cloned here previously."
        )

    # 3. Change the working directory
    os.chdir(project_path)
    print(f"✨ Changed working directory to: {os.getcwd()}")

    # 4. Configure Git Identity
    print("Configuring Git user identity...")
    subprocess.run(["git", "config", "--global", "user.name", user_name], check=True)
    subprocess.run(["git", "--global", "user.email", user_email], check=True)

    # 5. Optional: Pull latest changes to stay up to date
    print("Pulling latest changes from GitHub...")
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True)
    except subprocess.CalledProcessError:
        # Fallback if the default branch is 'master' instead of 'main'
        subprocess.run(["git", "pull", "origin", "master"], check=True)


if __name__ == "__main__":
    # --- CONFIGURATION ---
    REPO_NAME = "learn_cv"
    USER_NAME = "Peter Wen"
    USER_EMAIL = "petwen1027@gmail.com"
    # ---------------------

    setup_github_environment(REPO_NAME, USER_NAME, USER_EMAIL)
