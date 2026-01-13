import functools
import hashlib
import inspect
import os
import shutil
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import diskcache as dc  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    dc = None
import pandas as pd
import yaml
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

from mlebench.registry import Competition
from mlebench.utils import (
    authenticate_kaggle_api,
    extract,
    get_diff,
    get_logger,
    get_path_to_callable,
    is_empty,
    load_yaml,
)

logger = get_logger(__name__)
if dc is None:  # pragma: no cover
    cache: dict[Any, Any] = {}
else:
    cache = dc.Cache("cache", size_limit=2**26)  # 64 MB


def create_prepared_dir(competition: Competition) -> None:
    competition.public_dir.mkdir(exist_ok=True, parents=True)
    competition.private_dir.mkdir(exist_ok=True, parents=True)


def download_and_prepare_dataset(
    competition: Competition,
    keep_raw: bool = True,
    overwrite_checksums: bool = False,
    overwrite_leaderboard: bool = False,
    skip_verification: bool = False,
) -> None:
    """
    Creates a `public` and `private` directory for the competition using the `prepare_fn`,
    downloading the competition's dataset zip file and extracting it into `raw` if needed.
    """

    assert is_valid_prepare_fn(
        competition.prepare_fn
    ), f"Provided `prepare_fn` doesn't take arguments `raw`, `private` and `public`!"

    try:
        ensure_leaderboard_exists(competition, force=overwrite_leaderboard)
    except Exception as e:
        # Leaderboards are only required for medal thresholds/ranking; preparing the dataset
        # (public/private split + answers) can still proceed without it.
        logger.warning(f"Skipping leaderboard download for `{competition.id}`: {e}")

    competition_dir = competition.raw_dir.parent

    competition.raw_dir.mkdir(exist_ok=True, parents=True)
    create_prepared_dir(competition)

    zipfile: Path | None = None
    actual_zip_checksum: str | None = None

    # If raw data already exists, don't force a Kaggle download.
    if is_empty(competition.raw_dir):
        zipfile = download_dataset(
            competition_id=competition.kaggle_id,
            download_dir=competition_dir,
            force=False,
        )
    else:
        zip_files = sorted(competition_dir.glob("*.zip"))
        if len(zip_files) == 1:
            zipfile = zip_files[0]
        elif len(zip_files) > 1:
            logger.warning(
                f"Found multiple zip files in `{competition_dir}`; skipping zip verification and using raw data as-is."
            )

    if overwrite_checksums or not skip_verification:
        expected_checksums = None
        if competition.checksums.is_file() and not overwrite_checksums:
            expected_checksums = load_yaml(competition.checksums)

        if zipfile is not None and zipfile.is_file():
            logger.info(f"Generating checksum for `{zipfile}`...")
            zip_checksum_fn = get_checksum
            if (
                isinstance(expected_checksums, dict)
                and "zip" in expected_checksums
                and isinstance(expected_checksums["zip"], str)
                and _try_infer_checksum_algorithm(expected_checksums["zip"]) == "sha256"
            ):
                zip_checksum_fn = get_checksum_sha256
            actual_zip_checksum = zip_checksum_fn(zipfile)

            if (
                isinstance(expected_checksums, dict)
                and "zip" in expected_checksums
                and isinstance(expected_checksums["zip"], str)
            ):
                expected_zip_checksum = expected_checksums["zip"]

                if actual_zip_checksum != expected_zip_checksum:
                    raise ValueError(
                        f"Checksum for `{zipfile}` does not match the expected checksum! "
                        f"Expected `{expected_zip_checksum}` but got `{actual_zip_checksum}`."
                    )

                logger.info(f"Checksum for `{zipfile}` matches the expected checksum.")
        elif isinstance(expected_checksums, dict) and "zip" in expected_checksums:
            logger.warning(
                f"Checksums for `{competition.id}` expect a zip file, but none was found in `{competition_dir}`; skipping zip verification."
            )

    if is_empty(competition.raw_dir):
        if zipfile is None:
            raise FileNotFoundError(
                f"Raw data directory `{competition.raw_dir}` is empty and no zip file is available to extract."
            )
        logger.info(f"Extracting `{zipfile}` to `{competition.raw_dir}`...")
        extract(zipfile, competition.raw_dir, recursive=False)
        logger.info(f"Extracted `{zipfile}` to `{competition.raw_dir}` successfully.")

    if not is_dataset_prepared(competition) or overwrite_checksums:
        if competition.public_dir.parent.exists() and overwrite_checksums:
            logger.info(
                f"Removing the existing prepared data directory for `{competition.id}` since "
                "`overwrite_checksums` is set to `True`..."
            )
            shutil.rmtree(competition.public_dir.parent)
            create_prepared_dir(competition)

        logger.info(
            f"Preparing the dataset using `{competition.prepare_fn.__name__}` from "
            f"`{get_path_to_callable(competition.prepare_fn)}`..."
        )

        competition.prepare_fn(
            raw=competition.raw_dir,
            public=competition.public_dir,
            private=competition.private_dir,
        )

        logger.info(f"Data for competition `{competition.id}` prepared successfully.")

    with open(competition.public_dir / "description.md", "w") as f:
        f.write(competition.description)

    if overwrite_checksums or not skip_verification:
        expected_checksums = None
        if competition.checksums.is_file() and not overwrite_checksums:
            expected_checksums = load_yaml(competition.checksums)

        if not skip_verification and isinstance(expected_checksums, dict) and not overwrite_checksums:
            if "zip" in expected_checksums:
                # Standard MLE-bench schema: {zip, public, private}
                if zipfile is None or not zipfile.is_file():
                    logger.warning(
                        f"Checksums for `{competition.id}` expect a zip file, but none was found in `{competition_dir}`; skipping checksum verification."
                    )
                else:
                    zip_algo = _try_infer_checksum_algorithm(expected_checksums.get("zip")) or "md5"
                    zip_checksum_fn = get_checksum_sha256 if zip_algo == "sha256" else get_checksum
                    zip_checksum = (
                        actual_zip_checksum if actual_zip_checksum is not None else zip_checksum_fn(zipfile)
                    )

                    public_algo = _try_infer_checksum_algorithm(expected_checksums.get("public")) or zip_algo
                    private_algo = _try_infer_checksum_algorithm(expected_checksums.get("private")) or zip_algo

                    actual_checksums: dict[str, Any] = {
                        "zip": zip_checksum,
                        "public": generate_checksums(competition.public_dir, algorithm=public_algo),
                        "private": generate_checksums(competition.private_dir, algorithm=private_algo),
                    }

                    if actual_checksums != expected_checksums:
                        logger.error(f"Checksums do not match for `{competition.id}`!")
                        diff = get_diff(
                            actual_checksums,
                            expected_checksums,
                            fromfile="actual_checksums",
                            tofile="expected_checksums",
                        )
                        raise ValueError(f"Checksums do not match for `{competition.id}`!\n{diff}")
                    logger.info(
                        f"Checksums for files in `{competition_dir}` match the expected checksums."
                    )
            elif "raw" in expected_checksums:
                # Offline/local schema: {raw, public, private}
                raw_algo = _try_infer_checksum_algorithm(expected_checksums.get("raw")) or "md5"
                public_algo = _try_infer_checksum_algorithm(expected_checksums.get("public")) or raw_algo
                private_algo = _try_infer_checksum_algorithm(expected_checksums.get("private")) or raw_algo

                actual_checksums = {
                    "raw": generate_checksums(competition.raw_dir, algorithm=raw_algo),
                    "public": generate_checksums(competition.public_dir, algorithm=public_algo),
                    "private": generate_checksums(competition.private_dir, algorithm=private_algo),
                }

                if actual_checksums != expected_checksums:
                    logger.error(f"Checksums do not match for `{competition.id}`!")
                    diff = get_diff(
                        actual_checksums,
                        expected_checksums,
                        fromfile="actual_checksums",
                        tofile="expected_checksums",
                    )
                    raise ValueError(f"Checksums do not match for `{competition.id}`!\n{diff}")
                logger.info(
                    f"Checksums for files in `{competition_dir}` match the expected checksums."
                )
            else:
                # Legacy schema: checksums for raw files only.
                raw_algo = _try_infer_checksum_algorithm(expected_checksums) or "md5"
                raw_checksums = generate_checksums(competition.raw_dir, algorithm=raw_algo)
                if raw_checksums != expected_checksums:
                    logger.error(f"Raw checksums do not match for `{competition.id}`!")
                    diff = get_diff(
                        raw_checksums,
                        expected_checksums,
                        fromfile="actual_raw_checksums",
                        tofile="expected_raw_checksums",
                    )
                    raise ValueError(f"Raw checksums do not match for `{competition.id}`!\n{diff}")
                logger.info(
                    f"Raw checksums for files in `{competition.raw_dir}` match the expected checksums."
                )
        else:
            logger.info(f"Generating checksums for files in `{competition_dir}`...")

            # Default to MD5 for backward compatibility when writing checksums.
            actual_checksums = {
                "public": generate_checksums(competition.public_dir),
                "private": generate_checksums(competition.private_dir),
            }

            if zipfile is not None and zipfile.is_file() and actual_zip_checksum is not None:
                actual_checksums["zip"] = actual_zip_checksum
            elif zipfile is not None and zipfile.is_file():
                actual_checksums["zip"] = get_checksum(zipfile)
            else:
                # Local/offline datasets may not have a Kaggle zip; fall back to checksumming the raw CSVs.
                actual_checksums["raw"] = generate_checksums(competition.raw_dir)

            if not competition.checksums.is_file() or overwrite_checksums:
                with open(competition.checksums, "w") as file:
                    yaml.dump(actual_checksums, file, default_flow_style=False)

                logger.info(f"Checksums for `{competition.id}` saved to `{competition.checksums}`.")

    if not keep_raw:
        logger.info(f"Removing the raw data directory for `{competition.id}`...")
        shutil.rmtree(competition.raw_dir)

    assert competition.public_dir.is_dir(), f"Public data directory doesn't exist."
    assert competition.private_dir.is_dir(), f"Private data directory doesn't exist."
    assert not is_empty(competition.public_dir), f"Public data directory is empty!"
    assert not is_empty(competition.private_dir), f"Private data directory is empty!"


def is_dataset_prepared(competition: Competition, grading_only: bool = False) -> bool:
    """Checks if the competition has non-empty `public` and `private` directories with the expected files."""

    assert isinstance(
        competition, Competition
    ), f"Expected input to be of type `Competition` but got {type(competition)}."

    public = competition.public_dir
    private = competition.private_dir

    if not grading_only:
        if not public.is_dir():
            logger.warning("Public directory does not exist.")
            return False
        if is_empty(public):
            logger.warning("Public directory is empty.")
            return False

    if not private.is_dir():
        logger.warning("Private directory does not exist.")
        return False
    if is_empty(private):
        logger.warning("Private directory is empty.")
        return False

    if not competition.answers.is_file():
        logger.warning("Answers file does not exist.")
        return False

    if not competition.sample_submission.is_file() and not grading_only:
        logger.warning("Sample submission file does not exist.")
        return False

    return True


def is_api_exception(exception: Exception) -> bool:
    """
    Return True if the exception looks like a Kaggle API transient error.

    Kaggle's Python client has changed over time; older versions exposed
    `kaggle.rest.ApiException`, newer versions raise `requests.exceptions.HTTPError`
    (via kagglesdk). This helper must not hard-import optional modules.
    """
    try:
        # Older kaggle client.
        from kaggle.rest import ApiException  # type: ignore

        if isinstance(exception, ApiException):
            return True
    except Exception:
        pass

    try:
        from requests import HTTPError  # type: ignore

        if isinstance(exception, HTTPError):
            # Retry on common transient codes (429/5xx).
            status = getattr(getattr(exception, "response", None), "status_code", None)
            return status in (429, 500, 502, 503, 504)
    except Exception:
        pass

    return False


@retry(
    retry=retry_if_exception(is_api_exception),
    stop=stop_after_attempt(3),  # stop after 3 attempts
    wait=wait_fixed(5),  # wait 5 seconds between attempts
    reraise=True,
)
def download_dataset(
    competition_id: str,
    download_dir: Path,
    quiet: bool = False,
    force: bool = False,
) -> Path:
    """Downloads the competition data as a zip file using the Kaggle API and returns the path to the zip file."""

    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    logger.info(f"Downloading the dataset for `{competition_id}` to `{download_dir}`...")

    api = authenticate_kaggle_api()

    try:
        api.competition_download_files(
            competition=competition_id,
            path=download_dir,
            quiet=quiet,
            force=force,
        )
    except Exception as e:
        if _need_to_accept_rules(str(e)):
            logger.warning("You must accept the competition rules before downloading the dataset.")
            _prompt_user_to_accept_rules(competition_id)
            download_dataset(competition_id, download_dir, quiet, force)
        else:
            raise

    zip_files = list(download_dir.glob("*.zip"))

    assert (
        len(zip_files) == 1
    ), f"Expected to download a single zip file, but found {len(zip_files)} zip files."

    zip_file = zip_files[0]

    return zip_file


def _need_to_accept_rules(error_msg: str) -> bool:
    return "You must accept this competition" in error_msg


def _prompt_user_to_accept_rules(competition_id: str) -> None:
    response = input("Would you like to open the competition page in your browser now? (y/n): ")

    if response.lower() != "y":
        raise RuntimeError("You must accept the competition rules before downloading the dataset.")

    webbrowser.open(f"https://www.kaggle.com/c/{competition_id}/rules")
    input("Press Enter to continue after you have accepted the rules...")


def is_valid_prepare_fn(preparer_fn: Any) -> bool:
    """Checks if the `preparer_fn` takes three arguments: `raw`, `public` and `private`, in that order."""

    try:
        sig = inspect.signature(preparer_fn)
    except (TypeError, ValueError):
        return False

    actual_params = list(sig.parameters.keys())
    expected_params = ["raw", "public", "private"]

    return actual_params == expected_params


def generate_checksums(
    target_dir: Path,
    exts: Optional[list[str]] = None,
    exclude: Optional[list[Path]] = None,
    algorithm: str = "md5",
) -> dict:
    """
    Generate checksums for the files directly under the target directory with the specified extensions.

    Args:
        target_dir: directory to generate checksums for.
        exts: List of file extensions to generate checksums for.
        exclude: List of file paths to exclude from checksum generation.

    Returns:
        A dictionary of form file: checksum.
    """

    if exts is None:
        exts = ["csv", "json", "jsonl", "parquet", "bson"]

    if exclude is None:
        exclude = []

    checksums: dict[str, str] = {}
    if algorithm == "md5":
        checksum_fn = get_checksum
    elif algorithm == "sha256":
        checksum_fn = get_checksum_sha256
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm!r}")

    for ext in exts:
        fpaths = target_dir.glob(f"*.{ext}")

        for fpath in fpaths:
            if not fpath.is_file():
                continue  # skip dirs named like `my/dir.csv/`

            if fpath in exclude:
                continue

            checksums[fpath.name] = checksum_fn(fpath)

    return checksums


def get_last_modified(fpath: Path) -> datetime:
    """Return the last modified time of a file."""

    return datetime.fromtimestamp(fpath.stat().st_mtime)


def file_cache(fn: Callable) -> Callable:
    """A decorator that caches results of a function with a Path argument, invalidating the cache when the file is modified."""

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if not (len(params) == 1 and params[0].annotation is Path):
        raise NotImplementedError("Only functions with a single `Path` argument are supported.")

    # Use `functools.wraps` to preserve the function's metadata, like the name and docstring.
    # Query the cache, but with an additional `last_modified` argument in the key, which has the
    # side effect of invalidating the cache when the file is modified.
    @functools.wraps(fn)
    def wrapper(fpath: Path) -> Any:
        last_modified = get_last_modified(fpath)
        key = (fn.__name__, str(fpath), last_modified)

        if key not in cache:
            cache[key] = fn(fpath)

        return cache[key]

    return wrapper


@file_cache
def get_checksum(fpath: Path) -> str:
    """Compute MD5 checksum of a file."""

    assert fpath.is_file(), f"Expected a file at `{fpath}`, but it doesn't exist."

    hash_md5 = hashlib.md5()
    file_size = os.path.getsize(fpath)

    # only show progress bar for large files (> ~5 GB)
    show_progress = file_size > 5_000_000_000

    with open(fpath, "rb") as f:
        for chunk in tqdm(
            iter(lambda: f.read(4_096), b""),
            total=file_size // 4096,
            unit="B",
            unit_scale=True,
            disable=not show_progress,
        ):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


@file_cache
def get_checksum_sha256(fpath: Path) -> str:
    """Compute SHA-256 checksum of a file."""

    assert fpath.is_file(), f"Expected a file at `{fpath}`, but it doesn't exist."

    hash_sha256 = hashlib.sha256()
    file_size = os.path.getsize(fpath)

    # only show progress bar for large files (> ~5 GB)
    show_progress = file_size > 5_000_000_000

    with open(fpath, "rb") as f:
        for chunk in tqdm(
            iter(lambda: f.read(4_096), b""),
            total=file_size // 4096,
            unit="B",
            unit_scale=True,
            disable=not show_progress,
        ):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def _try_infer_checksum_algorithm(expected_checksums: Any) -> str | None:
    """
    Best-effort inference for expected checksum algorithm based on checksum string length.

    - MD5: 32 hex chars
    - SHA-256: 64 hex chars
    """

    if isinstance(expected_checksums, str):
        if len(expected_checksums) == 32:
            return "md5"
        if len(expected_checksums) == 64:
            return "sha256"
        return None

    if isinstance(expected_checksums, dict):
        for v in expected_checksums.values():
            inferred = _try_infer_checksum_algorithm(v)
            if inferred is not None:
                return inferred

    return None


def ensure_leaderboard_exists(competition: Competition, force: bool = False) -> Path:
    """
    Ensures the leaderboard for a given competition exists in the competition's
    directory, returning the path to it.
    If `force` is True, the leaderboard is downloaded using the Kaggle API.
    If `force` is `false`, if the leaderboard does not exist, an error is raised.
    """
    download_dir = competition.leaderboard.parent
    leaderboard_path = competition.leaderboard
    if not force:
        if leaderboard_path.exists():
            return leaderboard_path
        else:
            raise FileNotFoundError(
                f"Leaderboard not found locally for competition `{competition.id}`. Please flag this to the developers."
            )
    api = authenticate_kaggle_api()
    try:
        # Prefer fetching the full leaderboard (pagination) so medal thresholds are meaningful.
        # Tunable via env vars to avoid extremely large downloads.
        page_size = int(os.environ.get("MLEBENCH_LEADERBOARD_PAGE_SIZE", "200") or "200")
        max_pages_env = os.environ.get("MLEBENCH_LEADERBOARD_MAX_PAGES", "").strip()
        max_pages = int(max_pages_env) if max_pages_env else 0  # 0 = no cap

        from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

        submissions = []
        page_token: str | None = None
        seen_tokens: set[str | None] = set()

        with api.build_kaggle_client() as kaggle:
            page = 0
            while True:
                if page_token in seen_tokens:
                    raise RuntimeError("Leaderboard pagination token repeated; aborting to avoid infinite loop.")
                seen_tokens.add(page_token)

                req = ApiGetLeaderboardRequest()
                req.competition_name = competition.kaggle_id
                req.page_size = page_size
                if page_token:
                    req.page_token = page_token

                resp = kaggle.competitions.competition_api_client.get_leaderboard(req)
                page_subs = resp.submissions or []
                submissions.extend(page_subs)

                page += 1
                page_token = resp.next_page_token or None

                if max_pages and page >= max_pages:
                    logger.warning(
                        "Reached MLEBENCH_LEADERBOARD_MAX_PAGES=%s for `%s` (rows=%s).",
                        max_pages,
                        competition.id,
                        len(submissions),
                    )
                    break
                if not page_token:
                    break

        if not submissions:
            raise RuntimeError(f"Failed to download leaderboard for competition `{competition.id}` (no rows).")

        rows = [row.__dict__ for row in submissions]
        leaderboard_df = pd.DataFrame(rows)
        # Kaggle has changed these columns over time; drop if present.
        leaderboard_df.drop(columns=["teamNameNullable", "teamName"], inplace=True, errors="ignore")

        leaderboard_df.to_csv(leaderboard_path, index=False)
        logger.info(
            f"Downloaded leaderboard for competition `{competition.id}` to `{download_dir.relative_to(Path.cwd()) / 'leaderboard.csv'}`."
        )
        return leaderboard_path
    except Exception as e:
        # Fallback to the older (first-page only) behavior.
        logger.warning("Full leaderboard download failed for `%s` (%s); falling back to first page only.", competition.id, e)
        leaderboard = api.competition_leaderboard_view(competition=competition.kaggle_id)
        if leaderboard:
            leaderboard = [row.__dict__ for row in leaderboard]
            leaderboard_df = pd.DataFrame(leaderboard)
            leaderboard_df.drop(columns=["teamNameNullable", "teamName"], inplace=True, errors="ignore")
            leaderboard_df.to_csv(leaderboard_path, index=False)
            logger.info(
                f"Downloaded leaderboard for competition `{competition.id}` to `{download_dir.relative_to(Path.cwd()) / 'leaderboard.csv'}`."
            )
            return leaderboard_path
        raise RuntimeError(f"Failed to download leaderboard for competition `{competition.id}`.") from e


def get_leaderboard(competition: Competition) -> pd.DataFrame:
    leaderboard_path = competition.leaderboard
    assert (
        leaderboard_path.exists()
    ), f"Leaderboard not found locally for competition `{competition.id}`."
    try:
        with open(leaderboard_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise FileNotFoundError(
                f"Leaderboard for `{competition.id}` is a Git LFS pointer file. "
                "Run `git lfs pull` to fetch it (or skip leaderboard-based ranking)."
            )
    except UnicodeDecodeError:
        # If it's not UTF-8, let pandas handle / error below.
        pass
    leaderboard_df = pd.read_csv(leaderboard_path)
    if "score" not in leaderboard_df.columns:
        score_col = None
        for col in leaderboard_df.columns:
            if col.strip().lower().lstrip("_") == "score":
                score_col = col
                break
        if score_col is None:
            for col in leaderboard_df.columns:
                if "score" in col.strip().lower():
                    score_col = col
                    break
        if score_col is not None and score_col != "score":
            leaderboard_df = leaderboard_df.rename(columns={score_col: "score"})
    return leaderboard_df
