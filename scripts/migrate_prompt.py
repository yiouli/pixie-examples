#!/usr/bin/env python

import json
from pathlib import Path

from pixie.prompts.prompt import BaseUntypedPrompt
from pixie.prompts.storage import _FilePromptStorage


def main():
    # Target directory for new prompts
    target_dir = Path(".pixie/prompts")

    # Instantiate storage - it will load nothing since the directory is empty
    storage = _FilePromptStorage(str(target_dir))

    # Source directory for old prompts
    old_dir = Path(".pixie/prompts_old")

    # Process each old JSON file
    for json_file in old_dir.glob("*.json"):
        print(f"Processing {json_file.name}")

        # Load the old prompt data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract prompt ID from filename
        prompt_id = json_file.stem

        # Create BaseUntypedPrompt from the data
        prompt = BaseUntypedPrompt(
            id=prompt_id,
            versions=data["versions"],
            default_version_id=data["defaultVersionId"],
            variables_schema=data.get("variablesSchema"),
        )

        # Save using the storage
        storage.save(prompt)

        print(f"Migrated prompt: {prompt_id}")


if __name__ == "__main__":
    main()
