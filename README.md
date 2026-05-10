# astrbot-gpt-image

Generate images through OpenAI-compatible image generation and edit endpoints.

## Features

- Supports text-to-image and image-to-image generation.
- Supports model listing, size options, and multi-key failover.

## Installation

1. Clone or download this repository.
2. Copy the `gpt_image` directory into your AstrBot plugin directory.
3. Open the AstrBot plugin configuration page and fill in the required settings.
4. Restart AstrBot or reload the plugin.

## Usage

- Main command: `/gpt画图`
- Detailed command examples: see `gpt_image/README.md`

## Repository Structure

- `gpt_image/main.py`
- `gpt_image/_conf_schema.json`
- `gpt_image/metadata.yaml`
- `gpt_image/README.md`

## Notes

- Sensitive local API endpoints and keys have been replaced with placeholders where applicable.
- Runtime-specific local config files are not included.
