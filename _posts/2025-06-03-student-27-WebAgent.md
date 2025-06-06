---
layout: post
comments: true
title: Recent Developments in GUI Web Agents
author: Genglin Liu (Student 27), James Shiffer (Student 25)
date: 2025-06-03
---


> Web agents are a new class of agents that can interact with the web. They are able to navigate the web, search for information, and perform tasks. They are a type of multi-modal agent that can use text, images, and other modalities to interact with the web. Since 2024, we have seen a surge in the development of web agents, with many new agents being developed and released. In this blog post, we survey the recent developments in the field of web and particularly GUI agents, and provide a comprehensive overview of the state of the art. We review core benchmarks - WebArena, VisualWebArena, Mind2Web, and AssistantBench — that have enabled systematic measurement of these capabilities. We discuss the backbone vision-language models that power these agents, as well as the recent advancement in reasoning.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

Graphical User Interface (GUI) web agents are autonomous agents that interact with websites much like a human user – by clicking, typing, and reading on web pages. In recent years, especially since 2023/2024, there has been rapid progress in developing general-purpose GUI web agents powered by large (multimodal) language models (LLMs). These agents aim to follow high-level instructions (e.g. “Find the cheapest red jacket on an online store and add it to the cart”) and execute various tasks across different websites. 

A key challenge driving current research is reasoning capability: GUI agents must plan multi-step actions, handle dynamic web content, hop between different web pages, and sometimes recover from mistakes or unexpected outcomes. This report surveys recent developments in this area, emphasizing how new architectures and training methods have enhanced reasoning, with a special focus on e-commerce applications (a domain that is naitively multi-modal, and demands sophisticated multi-step reasoning, such as searching for products, comparing options, and completing purchases). We also contrast these GUI agents with earlier text-based web agents to highlight differences in reasoning approaches.


## Background: From Text-Based to Multi-modal Web Agents

Early “web agents” were often text-based, interfacing with the web via textual inputs/outputs or APIs rather than through a visual GUI. For example, a text-based agent might read the HTML or use a browser’s accessibility layer to get page text, then choose an action like following a link or submitting a form by ID. Notable early systems included OpenAI’s WebGPT (2021) [1] for web question-answering and various reinforcement learning (RL) agents on simplified web environments. These text-based agents relied on parsing textual content and often used search engine APIs or DOM trees. Their reasoning largely resembled traditional NLP tasks – e.g. retrieving relevant information or doing reading comprehension – and they did not need to reason about layout or visual elements.

By contrast, GUI web agents operate on the actual rendered web interface (like a human using a browser). This introduces new challenges and differences in reasoning.

The first one is *Perception and Grounding*: GUI agents must interpret the visual layout and GUI elements. They may receive a DOM tree or a rendered screenshot of the page. For instance, a task might involve recognizing a “red BUY button” or an image of a product. Text-only models struggle with such visual cues. Thus, GUI agents often incorporate vision (to understand images, colors, iconography) in addition to text. Reasoning for GUI agents often means linking instructions to the correct GUI element or image – a form of grounded reasoning not present in purely text agents.

The second one is *Action Space & Planning*: A text-based agent’s actions can be abstract (e.g. “go to URL” or “click link by name”), whereas a GUI agent deals with low-level events (mouse clicks, typing) on specific coordinates or element IDs. This requires more procedural reasoning: the agent must plan a sequence of GUI operations to achieve the goal. Often the agent must navigate through multiple pages or UI states, which demands long-horizon planning. For example, buying a product involves searching, filtering results, clicking the item, adding to cart, and possibly checkout – multiple steps that must be reasoned about and executed in order. Modern GUI agents leverage LLMs’ ability to plan by breaking down goals into sub-goals, sometimes explicitly writing out multi-step plans [2]. In comparison, text-based agents typically handle shorter-horizon tasks (like finding a specific fact on Wikipedia) with simpler sequential reasoning.

The third one is *Dynamic Interaction and Uncertainty*: Web GUIs are dynamic – pages can change or have pop-ups, and incorrect actions can lead the agent astray. GUI agents need robust error recovery strategies. This has led to new reasoning techniques like reflection and rollback. For example, an agent might try an action, realize it led to an irrelevant page, and then go back and try an alternative – a capability highlighted as essential in recent work [3]. Text agents, while they also handle some dynamic choices (e.g. picking the next link to click), typically operate in a more static information space and have less need for such UI-specific recovery reasoning.

GUI web agents require visual understanding, sequential decision-making, and error-aware reasoning that goes beyond what text-based agents historically needed. These differences have encouraged new research directions to equip GUI agents with better reasoning skills. Before diving into those developments, we next overview the benchmarks and environments that have been created to train and evaluate modern GUI web agents.


## Core Benchmarks

## References

[1] Nakano, Reiichiro, et al. "Webgpt: Browser-assisted question-answering with human feedback." arXiv preprint arXiv:2112.09332 (2021).

[2] Yang, Ke, et al. "Agentoccam: A simple yet strong baseline for llm-based web agents." arXiv preprint arXiv:2410.13825 (2024).

[3] Hu, Minda, et al. "WebCoT: Enhancing Web Agent Reasoning by Reconstructing Chain-of-Thought in Reflection, Branching, and Rollback." arXiv preprint arXiv:2505.20013 (2025).
