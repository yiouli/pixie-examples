# Pixie Examples

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1459772566528069715?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/YMNYu6Z3)

This repository contains a collection of example applications integrated with [**Pixie SDK**](https://github.com/yiouli/pixie-sdk-py) for interactive debugging.

## Get Started

> You can play with the demo site [here](https://gopixie.ai/?url=https://demo.yiouli.us/graphql) withou any setup.

### 1. Setup

Clone this repository:

Install `pixie-examples` python package:

```bash
pip install pixie-examples
```

Create a `.env` file with your API keys:

```ini
# .env
OPENAI_API_KEY=...
# Add other API keys as needed for specific examples
```

Start the Pixie server:

```bash
pixie
```

### 2. Debug with Web UI

Visit [gopixie.ai](https://gopixie.ai) to interact with and debug your applications through the web interface.

## Important Links

- [**Pixie SDK**](https://github.com/yiouli/pixie-sdk-py)
- [**Documentation**](https://yiouli.github.io/pixie-sdk-py/)
- [**Discord**](https://discord.gg/YMNYu6Z3)

## Examples Catelog

### Quickstart

- **Basic Example**: Simple hello world application to get started with Pixie SDK

### Pydantic AI Examples

- **Bank Support**: Multi-turn chatbot for banking customer support
- **Flight Booking**: Multi-agent system for flight booking
- **Question Graph**: Graph-based question answering system
- **SQL Generation**: Multi-step workflow for generating SQL queries
- **Structured Output**: Examples of structured data handling

### OpenAI Agents SDK Examples

- **Customer Service**: Multi-agent customer service system
- **Financial Research Agent**: Multi-step financial research workflow
- **LLM-as-a-Judge**: Evaluation and judging patterns
- **Routing**: Agent routing and handoffs

### LangChain Examples

- **Basic Agent**: Simple LangChain agent integration
- **Customer Support**: Customer support chatbot
- **Personal Assistant**: Multi-agent personal assistant
- **SQL Agent**: SQL query generation with LangChain

### LangGraph Examples

- **RAG System**: Retrieval-augmented generation with LangGraph
- **SQL Agent**: SQL agent built with LangGraph state machines
