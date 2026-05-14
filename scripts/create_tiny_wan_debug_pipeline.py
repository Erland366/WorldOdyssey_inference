"""Create tiny random-weight Wan Diffusers pipelines for fast debugging."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldodyssey_inference.tiny_wan import (
    RECIPES,
    RECIPE_ALIASES,
    build_tiny_wan_pipeline,
    load_source_tokenizer,
    push_folder,
    require_hf_token,
    resolve_recipe,
    resolve_repo_id,
    save_pipeline,
    verify_saved_pipeline,
    write_model_card,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="wan2.1-t2v-1.3b", help="Tiny Wan recipe name or alias.")
    parser.add_argument("--list-recipes", action="store_true", help="List supported recipe names and exit.")
    parser.add_argument("--source-model-id", default=None, help="Override the recipe source model id.")
    parser.add_argument("--tokenizer-subfolder", default=None, help="Override the recipe tokenizer subfolder.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--repo-name", default=None)
    return parser.parse_args()


def print_recipes() -> None:
    print("Recipes:")
    for name, recipe in sorted(RECIPES.items()):
        print(f"  {name}: {recipe.source_model_id} [{recipe.tokenizer_subfolder}]")
    print("Aliases:")
    for alias, name in sorted(RECIPE_ALIASES.items()):
        print(f"  {alias} -> {name}")


def main() -> None:
    args = parse_args()
    if args.list_recipes:
        print_recipes()
        return

    load_dotenv(dotenv_path=Path.cwd() / ".env")

    recipe = resolve_recipe(args.recipe)
    if args.source_model_id is not None:
        recipe = replace(recipe, source_model_id=args.source_model_id)
    if args.tokenizer_subfolder is not None:
        recipe = replace(recipe, tokenizer_subfolder=args.tokenizer_subfolder)

    output_dir = args.output_dir or recipe.default_output_dir
    repo_name = args.repo_name or recipe.default_repo_name
    local_card_repo_id = args.repo_id or f"YOUR_HF_USERNAME/{repo_name}"

    tokenizer = load_source_tokenizer(recipe)
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=tokenizer, seed=args.seed)
    save_pipeline(
        pipeline=pipeline,
        output_dir=output_dir,
        overwrite=args.overwrite,
        recipe=recipe,
        repo_id=local_card_repo_id,
    )
    verification_message = verify_saved_pipeline(output_dir=output_dir, recipe=recipe)
    print(f"Saved {recipe.name} tiny Wan debug pipeline to {output_dir}")
    print(verification_message)

    if args.push:
        token = require_hf_token()
        repo_id = resolve_repo_id(repo_id=args.repo_id, repo_name=repo_name, token=token)
        write_model_card(output_dir=output_dir, recipe=recipe, repo_id=repo_id)
        push_folder(output_dir=output_dir, repo_id=repo_id, token=token)
        print(f"Pushed public Hugging Face model repo: {repo_id}")


if __name__ == "__main__":
    main()
