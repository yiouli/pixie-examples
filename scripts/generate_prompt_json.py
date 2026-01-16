#!/usr/bin/env python

import asyncio
import inspect
import os
import traceback

import openai

from pixie import initialize_prompt_storage, StorageBackedPrompt
from scripts.prompts import prompts


openai.api_key = "sk-your-api-key"


async def main():
    dir = os.path.join("pixie1/prompts")
    initialize_prompt_storage(dir)
    print(f"Initialized prompt storage at: {dir}")

    import importlib.util

    # Assuming StorageBackedPrompt is imported from pixie or wherever it's defined

    def find_storage_backed_prompts_in_examples():
        examples_dir = "examples"
        pmts: list[StorageBackedPrompt] = []

        for root, _dirs, files in os.walk(examples_dir):
            for file in files:
                if file.endswith(".py"):
                    module_path = os.path.join(root, file)
                    module_name = module_path.replace(os.sep, ".").replace(".py", "")

                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, module_path
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            for name, obj in inspect.getmembers(module):
                                if isinstance(obj, StorageBackedPrompt):
                                    print(
                                        f"Found in {module}: {name} = {obj.id}<{obj.variables_definition}"
                                    )
                                    pmts.append(obj)
                    except Exception as e:
                        print(f"Error loading module {module_name}: {e}")

        return pmts

    # Find all StorageBackedPrompt instances
    found_prompts = find_storage_backed_prompts_in_examples()
    # Process each prompt from the prompts dictionary
    created_count = 0
    for prompt in found_prompts:
        try:
            content = prompts[prompt.id]
            prompt.append_version("v0", content, set_as_default=True)
            print(f"Created prompt: {prompt.id}")
            created_count += 1
        except Exception as e:
            traceback.print_exc()
            print(f"ERROR creating prompt '{prompt.id}': {e}")

    print(f"\nDone! Generated {created_count} prompt JSON files.")
    print(f"Check the directory: {dir}")


if __name__ == "__main__":
    asyncio.run(main())
