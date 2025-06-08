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

Graphical User Interface (GUI) web agents are autonomous agents that interact with websites much like a human user – by clicking, typing, and reading on web pages. In recent years, especially since 2023/2024, there has been rapid progress in developing general-purpose GUI web agents powered by large (multimodal) language models (LLMs). These agents aim to follow high-level instructions (e.g. "Find the cheapest red jacket on an online store and add it to the cart") and execute various tasks across different websites. 

A key challenge driving current research is reasoning capability: GUI agents must plan multi-step actions, handle dynamic web content, hop between different web pages, and sometimes recover from mistakes or unexpected outcomes. This report surveys recent developments in this area, emphasizing how new architectures and training methods have enhanced reasoning, with a special focus on e-commerce applications (a domain that is naitively multi-modal, and demands sophisticated multi-step reasoning, such as searching for products, comparing options, and completing purchases). We also contrast these GUI agents with earlier text-based web agents to highlight differences in reasoning approaches.


## Background: From Text-Based to Multi-modal Web Agents

Early "web agents" were often text-based, interfacing with the web via textual inputs/outputs or APIs rather than through a visual GUI. For example, a text-based agent might read the HTML or use a browser’s accessibility layer to get page text, then choose an action like following a link or submitting a form by ID. Notable early systems included OpenAI’s WebGPT (2021) [1] for web question-answering and various reinforcement learning (RL) agents on simplified web environments. These text-based agents relied on parsing textual content and often used search engine APIs or DOM trees. Their reasoning largely resembled traditional NLP tasks – e.g. retrieving relevant information or doing reading comprehension – and they did not need to reason about layout or visual elements.

By contrast, GUI web agents operate on the actual rendered web interface (like a human using a browser). This introduces new challenges and differences in reasoning.

The first one is *Perception and Grounding*: GUI agents must interpret the visual layout and GUI elements. They may receive a DOM tree or a rendered screenshot of the page. For instance, a task might involve recognizing a "red BUY button" or an image of a product. Text-only models struggle with such visual cues. Thus, GUI agents often incorporate vision (to understand images, colors, iconography) in addition to text. Reasoning for GUI agents often means linking instructions to the correct GUI element or image – a form of grounded reasoning not present in purely text agents.

The second one is *Action Space & Planning*: A text-based agent’s actions can be abstract (e.g. "go to URL" or "click link by name"), whereas a GUI agent deals with low-level events (mouse clicks, typing) on specific coordinates or element IDs. This requires more procedural reasoning: the agent must plan a sequence of GUI operations to achieve the goal. Often the agent must navigate through multiple pages or UI states, which demands long-horizon planning. For example, buying a product involves searching, filtering results, clicking the item, adding to cart, and possibly checkout – multiple steps that must be reasoned about and executed in order. Modern GUI agents leverage LLMs’ ability to plan by breaking down goals into sub-goals, sometimes explicitly writing out multi-step plans [2]. In comparison, text-based agents typically handle shorter-horizon tasks (like finding a specific fact on Wikipedia) with simpler sequential reasoning.

The third one is *Dynamic Interaction and Uncertainty*: Web GUIs are dynamic – pages can change or have pop-ups, and incorrect actions can lead the agent astray. GUI agents need robust error recovery strategies. This has led to new reasoning techniques like reflection and rollback. For example, an agent might try an action, realize it led to an irrelevant page, and then go back and try an alternative – a capability highlighted as essential in recent work [3]. Text agents, while they also handle some dynamic choices (e.g. picking the next link to click), typically operate in a more static information space and have less need for such UI-specific recovery reasoning.

GUI web agents require visual understanding, sequential decision-making, and error-aware reasoning that goes beyond what text-based agents historically needed. These differences have encouraged new research directions to equip GUI agents with better reasoning skills. Before diving into those developments, we next overview the benchmarks and environments that have been created to train and evaluate modern GUI web agents.


## Core Benchmarks

In recent years, several benchmarks and agent frameworks have been proposed to evaluate and advance autonomous agents that can interact with websites through a graphical user interface (GUI). These systems combine natural language understanding, web page vision/DOM parsing, and action execution (clicking, typing, etc.) to follow instructions on real or simulated websites. In this section, we review five notable recent projects – VisualWebArena, Mind2Web, Online-Mind2Web, WebVoyager, and RealWebAssist – focusing on their core ideas, evaluation methodologies (tasks, datasets, metrics), key results, and the strengths/limitations of each approach.

### VisualWebArena 

VisualWebArena (VWA) is a benchmark designed to test multimodal web agents on realistic tasks that require visual understanding of web content [4]. Prior web agent benchmarks mostly used text-only information, but VWA includes images and graphical elements as integral parts of the tasks. The goal is to evaluate an agent’s ability to process image-text inputs, interpret natural language instructions, and execute actions on websites to accomplish user-defined objectives. This benchmark builds on the WebArena framework (a realistic web environment) by introducing new tasks where visual content is crucial (e.g. identifying items by images or interpreting graphical page elements).

VWA comprises 910 diverse tasks in three web domains: an online Classifieds site (newly created with real-world data), an e-Shopping site, and a Reddit-style forum. Tasks range from purchasing specific products (which may require recognizing product images) to navigating forums with embedded media. Agents interact with webpages that are self-hosted for consistency. Success is measured by whether the agent achieves the correct end state on the website (functional task completion). Human performance on these tasks is about 88–89% success, indicating the tasks are feasible for people but non-trivial. An example task is: "Find a listing for a red convertible car on the classifieds site and report its price" – requires the agent to visually identify the car’s image/color on the page (beyond just text).

The benchmark supports various agent types. The authors evaluated state-of-the-art large language model (LLM) based agents, including text-only agents and vision-enabled agents (e.g. GPT-4 with vision, open-source vision-language models). Agents observe either a DOM-derived accessibility tree (textual interface) or the rendered webpage image (for vision models), sometimes augmented with image captions (generated by a tool like BLIP-2) for text models. Agents produce actions such as clicking UI elements or entering text. Some experiments also used a Set-of-Mark (SoM) [5] approach – a method to highlight possible interactive elements for the agent to consider – to aid decision making

**Key Results**: Text-Only vs Multimodal: A GPT-4 based agent using only text (no vision) achieved under 10% success on VWA tasks (≈7% in one setting). In contrast, GPT-4V (Vision-enabled GPT-4) achieved roughly 15% overall success, more than doubling the success rate by leveraging page images. Even augmenting a text agent with image captions improved success to ~12.8%, but fell short of a true vision model. This underscores that visual context is often critical: purely text-based agents miss cues that are obvious to multimodal agents (e.g. recognizing a product from its photo). The evaluation showed text-only LLM agents often fail when tasks involve identifying visual attributes (colors, shapes, etc.) or non-textual cues. VisualWebArena is the first benchmark to focus on visually-grounded web tasks in a realistic setting. It provides a controlled environment with reproducible websites, enabling rigorous comparison of agents. It highlights the importance of vision in web agents: many real-world web tasks (shopping for an item of a certain style, recognizing icons/buttons, etc.) cannot be solved by text parsing alone.

### Mind2Web

Mind2Web is introduced as the first large-scale dataset for developing and evaluating a "generalist" web agent – one that can follow natural language instructions to complete complex tasks on any website [6]. Prior benchmarks often used either simulated web pages or a limited set of websites/tasks, which risked overfitting. Mind2Web instead emphasizes breadth and diversity: it contains over 2,000 tasks collected from 137 real-world websites spanning 31 domains (e-commerce, social media, travel, etc.). Each task is open-ended (expressed in natural language) and comes with a human-crowdsourced action sequence demonstrating how to complete it, providing both a goal specification and a ground-truth solution path.

### Online-Mind2Web

Online-Mind2Web is a follow-up benchmark that addresses a critical question: Are web agents as competent as some benchmarks suggest, or are we seeing an illusion of progress? [7] The authors of this study argue that previously reported results might be overly optimistic due to evaluation on static or simplified environments. Online-Mind2Web instead evaluates agents in a live web setting, where they must perform tasks on actual, up-to-date websites – thus exposing them to real-world variability and unpredictability. It consists of 300 diverse tasks spanning 136 popular websites, collected to approximate how real users would instruct an agent online. This includes tasks across domains like shopping, travel, social media, etc., similar in spirit to Mind2Web’s diversity but executed on the real web.

The authors developed an LLM-as-a-Judge system called WebJudge to automatically evaluate whether an agent succeeded in a given task, with minimal human labor. WebJudge uses an LLM to analyze the agent’s action history and the resulting web states (with key screenshots) to decide if the goal was achieved. This method achieved about 85% agreement with human evaluators on success judgments, substantially higher consistency than prior automatic metrics, substantially higher consistency than prior automatic metrics. 


### WebVoyager

WebVoyager [8] has a set of tasks that are open-ended user instructions that require multi-step interactions. For example, tasks might include "Book a hotel room in New York for next weekend on Booking.com" or "Find and compare the prices of two specific smartphones on an e-commerce site". The tasks are meant to reflect practical web goals (searching, form-filling, navigating multi-page flows, etc.) and often require combining information gathering with action execution. Because evaluating success on such open tasks can be tricky, WebVoyager’s authors used an automatic evaluation protocol leveraging GPT-4V (similar in spirit to WebJudge). They had GPT-4V examine the agent’s outcome (e.g., final page or sequence of pages visited) and compare it to the intended goal, achieving about 85% agreement with human judgment of success.

Its benchmark and results validated that multimodal agents can substantially outperform text-only agents on real-world websites. Another strength is the introduction of an automatic evaluation metric with GPT-4V to reduce reliance on slow human evaluation

### RealWebAssist

RealWebAssist [9] is a benchmark aimed at long-horizon, sequential web assistance with real users. Unlike one-shot tasks (e.g. "buy X from Amazon"), this benchmark considers scenarios where a user interacts with an AI assistant over an extended session comprising multiple interdependent instructions. The key idea is to simulate a personal web assistant that can handle an evolving to-do list or goal that unfolds through conversation. RealWebAssist addresses several real-world complexities not covered in prior benchmarks. It extends the challenge to interactive, personalized assistance, highlighting the significant gap between current AI capabilities and the full vision of a personal web assistant that can handle ambiguous, evolving user needs over long sessions.

For example, a user might start with "I need a gift for my mother’s birthday", then later add "She likes gardening" – the agent must adjust its plan. The benchmark includes such evolving goals. The agent might need to learn or keep track of a particular user’s preferences/routines. E.g., "Order my usual Friday takeout" implies knowing what "usual" means for that user. Instructions come from real users (not annotators inventing them), so they reflect personal language quirks and context assumptions.

As with other GUI agents, the assistant must interface with actual web pages (click buttons, fill forms). RealWebAssist emphasizes aligning instructions with the correct GUI actions – e.g., if the user says "open the second result," the agent must translate that to clicking the second item on the page.

The benchmark consists of sequential instruction episodes collected from real users. For evaluation, the authors consider the following evaluation metrics:
- Task success rate: Measures whether the web agent successfully executes all instructions in a given task sequence.
- Average progress: Calculates what percentage of instructions the agent completes correctly before making its first mistake in a task.
- Step success rate: Evaluates the agent's performance in a controlled setting where it only needs to handle one instruction at a time, with all previous steps assumed to be completed successfully.

## Backbone Vision-Language Models and UI Grounding

Multimodal LLMs have become the core enablers of GUI agents. Since 2024, a new generation of such models has been adapted to interact with digital interfaces using vision and language. Notably, closed-source giants like OpenAI’s GPT‑4 and Google’s Gemini [17] are pushing capability boundaries. OpenAI's developed a Computer-Using Agent (CUA) to drive an autonomous GUI agent named **Operator** [18]. This system can visually perceive screens and perform mouse/keyboard actions without specialized APIs. Anthropic’s Claude 3.5 [19] has also gained a “Computer Use” capability, enabling it to "view" a virtual screen and emulate mouse/keyboard interactions. Meanwhile, the community has introduced strong open-source multimodal models to democratize GUI agents. Notably, Alibaba’s Qwen-2.5-VL [20] has delivered impressive visual understanding – e.g. it can handle very large image inputs and precise object localization. Similarly, the ChatGLM family has spawned AutoGLM [21], a specialized multimodal agent model. AutoGLM augments a ChatGLM-based LLM with vision and is trained via reinforcement learning for GUI tasks.


### UI Grounding - Aligning Perception and Action

Achieving fluent GUI automation requires “grounding” the model in the actual UI – i.e. linking the model’s linguistic plans to on-screen elements and executable actions. Recent research has made substantial progress in perception, UI understanding, and action grounding for GUI agents.


## Architectures and Methods: Enhancing Reasoning in GUI Agents

Developing a general-purpose GUI web agent requires innovation on multiple fronts: how the agent’s policy is represented (LLM-based, reinforcement learned, etc.), how it integrates vision and text, and how it plans and reasons about its next actions. Recent works have explored a spectrum of approaches:

### Prompt-Based Planning and Chain-of-Thought

Large Language Models, especially since GPT-4 [10] and their open-source equivalents, have become the "brains" of many web agents. Given an instruction and the current page (as text or description of GUI), the LLM can be prompted to output an action (e.g. click("Add to Cart")). Early attempts simply used zero-shot or few-shot prompting with GPT-4, but these often lacked deep reasoning for multi-step tasks. To address this, researchers introduced chain-of-thought (CoT) strategies for web agents: the model is encouraged to generate a step-by-step rationale before choosing each action. This was inspired by the ReAct paradigm [11], which interleaves reasoning text ("I need to find a search bar on the page...") with explicit actions. By reasoning explicitly, the LLM can plan several steps ahead or consider contingencies.

A recent development in 2025 is **WebCoT [3]**, which identifies key reasoning skills for web agents and trains models to use them. WebCoT emphasizes reflection & lookahead, branching, and rollback as three crucial abilities. Instead of relying on an LLM to implicitly learn these, they reconstruct training trajectories with rich rationales that demonstrate, for example, branching (considering multiple possible paths) and rollback (recovering from a bad click) in the chain-of-thought. By fine-tuning an LLM on these augmented trajectories, WebCoT achieved notable gains on WebVoyager, and on tasks like Mind2Web-live and web search QA. 

Similarly, other works have looked at incorporating planning abilities directly into the model’s action space. **AgentOccam [2]** by Amazon researchers is one such approach: it introduces special actions like branch and prune which allow the agent to explicitly create and manage a tree of sub-plans. For example, an e-commerce task might require information from two different pages; the agent can branch its plan into two parallel subgoals, gather results, then prune away completed branches. By simplifying the action space (removing distracting or seldom-used actions) and adding these planning operations, AgentOccam achieved a strong performance baseline on the WebArena benchmark. 

Another notable result of incorporating better reasoning came from LASER (2024), which tackled WebShop tasks. LASER reframed the web task for the LLM as an explicit state-space exploration problem [12]. Instead of having the LLM treat every step as unrelated to the next, LASER defines distinct states (e.g. “search results page” or “product page”) and allowed actions from each state, enabling the model to backtrack and explore alternatives. This approach acknowledged that an LLM agent may make a mistake, but it can recover by returning to a previous state and trying a different path (much like a human clicking the back button and trying the next link). The impact of this reasoning-aware formulation was striking on e-commerce tasks – LASER nearly closed the gap with human performance on WebShop [13].

### Reinforcement Learning

While prompting and supervised fine-tuning have seen success, another thread of research focuses on learning from experience to improve GUI agents. The idea is that by actually trial-and-error interacting with the environment, an agent can learn a policy that is more robust than one purely imitated from limited data. However, applying RL to large language model-based agents is non-trivial (due to huge action spaces, long horizons, and the expense of running many trials). Recent work has made progress on this front:

**WebAgent-R1 (2025)** [14] is an end-to-end RL framework that achieved state-of-the-art performance on WebArena by carefully adapting RL algorithms to LLM agents. The authors designed a reward function that encourages the agent to complete the task at hand while also exploring the web. They also introduced a method to handle the sparse and delayed reward signal present in web navigation tasks. One innovation in WebAgent-R1 is a dynamic context compression mechanism: as an episode proceeds through dozens of pages, the text observation can become thousands of tokens long (exceeding model limits). WebAgent-R1 dynamically truncates or compresses less relevant context from earlier steps to keep the input size manageable. Another contribution is extending GRPO [15] to multi-turn settings (called M-GRPO) and using asynchronous parallel browsing to collect experiences faster.

Another angle of learning from experience is self-improvement via trajectory relabeling or self-play. Researchers have explored letting an LLM agent run on many tasks, then analyzing its failures and successes to generate additional training data [16]. The agent would attempt tasks, then (using the LLM’s own evaluation or a heuristic) identify where it went wrong, and synthesize corrected trajectories or critiques. These synthetic trajectories, added to a finetuning dataset, boosted the task completion rate by ~31% over the base model on WebArena. Such self-refinement approaches are appealing because they mitigate the need for large human-labeled datasets – the agent essentially learns from its mistakes. However, they also require careful validation to ensure the agent doesn’t reinforce bad behaviors. 



## Replication Study: Evaluating Qwen-2.5-VL as a GUI Agent


## Conclusion


## References

[1] Nakano, Reiichiro, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse et al. "Webgpt: Browser-assisted question-answering with human feedback." arXiv preprint arXiv:2112.09332 (2021).

[2] Yang, Ke, Yao Liu, Sapana Chaudhary, Rasool Fakoor, Pratik Chaudhari, George Karypis, and Huzefa Rangwala. "Agentoccam: A simple yet strong baseline for llm-based web agents." arXiv preprint arXiv:2410.13825 (2024).

[3] Hu, Minda, Tianqing Fang, Jianshu Zhang, Junyu Ma, Zhisong Zhang, Jingyan Zhou, Hongming Zhang, Haitao Mi, Dong Yu, and Irwin King. "WebCoT: Enhancing Web Agent Reasoning by Reconstructing Chain-of-Thought in Reflection, Branching, and Rollback." arXiv preprint arXiv:2505.20013 (2025).

[4] Koh, Jing Yu, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. "Visualwebarena: Evaluating multimodal agents on realistic visual web tasks." arXiv preprint arXiv:2401.13649 (2024).

[5] Yang, Jianwei, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. "Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v." arXiv preprint arXiv:2310.11441 (2023).

[6] Deng, Xiang, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. "Mind2web: Towards a generalist agent for the web." Advances in Neural Information Processing Systems 36 (2023): 28091-28114.

[7] Xue, Tianci, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. "An illusion of progress? assessing the current state of web agents." arXiv preprint arXiv:2504.01382 (2025).

[8] He, Hongliang, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. "WebVoyager: Building an end-to-end web agent with large multimodal models." arXiv preprint arXiv:2401.13919 (2024).

[9] Ye, Suyu, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, and Tianmin Shu. "Realwebassist: A benchmark for long-horizon web assistance with real-world users." arXiv preprint arXiv:2504.10445 (2025).

[10] Achiam, Josh, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida et al. "Gpt-4 technical report." arXiv preprint arXiv:2303.08774 (2023).

[11] Yao, Shunyu, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. "React: Synergizing reasoning and acting in language models." In International Conference on Learning Representations (ICLR). 2023.

[12] Ma, Kaixin, Hongming Zhang, Hongwei Wang, Xiaoman Pan, Wenhao Yu, and Dong Yu. "Laser: Llm agent with state-space exploration for web navigation." arXiv preprint arXiv:2309.08172 (2023).

[13] Yao, Shunyu, Howard Chen, John Yang, and Karthik Narasimhan. "Webshop: Towards scalable real-world web interaction with grounded language agents." Advances in Neural Information Processing Systems 35 (2022): 20744-20757.

[14] Wei, Zhepei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu et al. "WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning." arXiv preprint arXiv:2505.16421 (2025).

[15] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

[16] Patel, Ajay, Markus Hofmarcher, Claudiu Leoveanu-Condrei, Marius-Constantin Dinu, Chris Callison-Burch, and Sepp Hochreiter. "Large language models can self-improve at web agent tasks." arXiv preprint arXiv:2405.20309 (2024).

[17] Team, Gemini, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk et al. "Gemini: a family of highly capable multimodal models." arXiv preprint arXiv:2312.11805 (2023).

[18] OpenAI. 2024. Introducing Operator. https://openai.com/index/introducing-operator/

[19] Anthropic. 2024. Anthropic's new Claude 3.5 Sonnet model is now available. https://www.anthropic.com/news/claude-3-5-sonnet

[20] Bai, Shuai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang et al. "Qwen2. 5-vl technical report." arXiv preprint arXiv:2502.13923 (2025).

[21] Liu, Xiao, Bo Qin, Dongzhu Liang, Guang Dong, Hanyu Lai, Hanchen Zhang, Hanlin Zhao et al. "Autoglm: Autonomous foundation agents for guis." arXiv preprint arXiv:2411.00820 (2024).

