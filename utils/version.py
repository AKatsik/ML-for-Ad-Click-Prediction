""" Definig the version of the ML pipeline. """


import subprocess

def get_git_version():
    """Returns the latest Git tag or commit hash as the version."""
    try:
        return subprocess.check_output(
            [
                "git"
                , "describe"
                , "--tags"
                , "--always"
            ]
            , stderr=subprocess.DEVNULL).strip().decode()

    except subprocess.CalledProcessError:#
        # If Git command failed (e.g., repo has no tags/commits)
        return "unknown"

    except FileNotFoundError:
        # If Git is not installed in the environment
        return "git_not_installed"

# Optional: Define __version__ so other scripts can import it
__version__ = get_git_version()
