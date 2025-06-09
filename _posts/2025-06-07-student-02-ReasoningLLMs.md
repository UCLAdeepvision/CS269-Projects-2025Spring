---
layout: post
comments: true
title: Balancing Reasoning and Efficiency in LLMs - Insights from Qwen3-8B
author: Renliang Sun
date: 2025-06-07
---


> Recent advances in large language models (LLMs) have led to remarkable improvements in reasoning abilities. However, excessive reasoning can incur high computational costs and sometimes harm accuracy on simple tasks. In this report, we trace the evolution of LLM reasoning techniques from early prompting to built-in reasoning modes in state-of-the-art models like Qwen3. We conduct experiments on Qwen3-8B to evaluate the impact of different reasoning strategies, including truncated reasoning, self-consistency voting, and no-thinking prompts. Our findings highlight the trade-offs between reasoning depth and efficiency, and suggest that dynamic reasoning control is essential to optimize LLM performance across tasks.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

In the context of large language models (LLMs), reasoning ability encompasses far more than pattern matching or word prediction. It refers to an artificially intelligent system's capacity to analyze information, draw logical conclusions, consider context and nuance, and solve problems through multi-step reasoning [1].

In the early stages of large language model development, LLMs like GPT-3.5 performed well in terms of language fluency but often exhibited biases in tasks requiring reasoning, such as counting the number of characters in a string or performing multi-step mathematical operations. These models primarily operate in what cognitive psychology refers to as “System 1” mode - rapid and intuitive, but lacking the deliberate logical planning required for “System 2” reasoning mode [2]. As a result, they frequently make simple logical errors or confidently provide incorrect answers that humans could avoid through basic reasoning.

Enhancing the reasoning capabilities of LLMs has become an important and rapidly evolving research objective, as many real-world applications—including solving math problems and understanding complex instructions—require some form of logical reasoning [3]. Models with strong reasoning capabilities can break down complex problems into multiple steps and explain their answers, significantly enhancing their reliability in decision support, automation, and knowledge-intensive tasks. In recent years, researchers have discovered that LLMs do possess latent reasoning capabilities, and through appropriate techniques—including pre-training and post-training—we can significantly improve their performance in reasoning-intensive tasks [4]. The field has evolved from simple prompting techniques to complex post-training and architectural adjustments, enabling models to develop an “internal reasoning capability. The remainder of this paper traces this evolutionary journey, from early prompting methods such as chain-of-thought prompts to the “thinking patterns” integrated into cutting-edge models like Qwen3 [5], and explores current research progress, including new experiments on the open-source model Qwen3-8B.

## Early Prompting Methods

A major breakthrough in the study of reasoning ability in 2022 was the discovery of the Chain-of-Thought (CoT) prompting methodology [6]. Unlike directly asking the model to output an answer, the CoT prompting methodology requires the model to generate a sequence of intermediate reasoning steps before arriving at the final answer. The experimental results are remarkable: for example, Wei et al.'s research shows that in the GSM8K mathematical benchmark, the PaLM-540B model achieved an accuracy rate of 57% after using CoT prompts, while the same model only achieved an accuracy rate of about 18% without using CoT prompts, which is a huge improvement. In other words, prompting the model to explain its reasoning process activated its latent “System 2” abilities.

Another closely related concept is the use of “scratchpads” or scratchpad prompts [7]. These are essentially explicit workspaces where models perform intermediate computations. In practice, scratchpad prompts are very similar to chain-of-reasoning: models are prompted to generate reasoning steps before giving the final answer, effectively allowing them to think. Their core idea is the same, which is to prevent models from jumping directly to conclusions by encouraging them to generate reasoning tokens before arriving at an answer.

However, a simple reasoning chain can still be flawed. LLMs may generate a seemingly reasonable reasoning chain but ultimately arrive at an incorrect answer, which could be due to computational errors or incorrect assumptions in the reasoning steps. This has motivated the development of new decoding strategies to better leverage multiple reasoning paths. One such method is self-consistency decoding [8]. Unlike directly adopting the model's first answer, self-consistency decoding runs the model multiple times to answer the same question and then selects the answer that appears most frequently across these paths. The core principle is: if multiple independent reasoning chains arrive at the same answer, that answer is more likely to be correct. This simple voting mechanism has led to a significant improvement in accuracy, with answer accuracy on the GSM8K dataset improving by +17.9% compared to a single reasoning chain.

​​At the same time, researchers explored prompting strategies to help models decompose problems themselves. One such method is the Least-of-Most prompting approach: it explicitly trains or prompts the model to first solve simpler subproblems, then gradually transition to the original complex problem [9]. Unlike a single chain of reasoning, the model first generates or obtains a solution to a simpler related problem, then uses that result to solve a slightly more difficult problem, and so on—progressing step by step from “least” complex to “most” complex.

Overall, these early prompting methods have demonstrated that the reasoning performance of LLMs can be significantly improved through prompt engineering alone. These methods do not require modification of the model's weights, but rather enable more intelligent utilization of the model during reasoning. The success of these methods indicates that reasoning is an emerging capability of LLMs, prompting researchers to explore training and architectural innovations to directly incorporate better reasoning capabilities into the models.

## Reasoning in Advanced LLMs

Although prompt-based methods have brought about rapid improvements in reasoning capabilities, the latest models have begun to incorporate reasoning enhancements in a more systematic manner. In 2024, OpenAI released an experimental model codenamed “o1”, which is essentially a further fine-tuned version of GPT-4 through an internal chained reasoning mechanism [10]. During training, o1 was rewarded for “thinking before answering”, meaning that before providing the final answer, it internally generated a series of reasoning steps. As a result, the model outperformed the base version of GPT-4 in a series of tasks requiring extensive reasoning. For example, OpenAI claims that o1 can rank among the top 500 human participants in the AIME exam and achieve scores surpassing humans on graduate-level scientific problems.

Around the same time, Google DeepMind's Gemini model began to incorporate reasoning as a core feature [11]. For example, Gemini 2.0 Flash is an experimental model capable of brief internal reasoning, while Gemini 2.5 Pro was explicitly designed for advanced reasoning. Google is incorporating the concept of “built-in reasoning mode” into all future models—that is, the ability to automatically perform complex multi-step analyses while still being able to quickly process simple queries. The model will dynamically determine when to enable its more powerful “System 2” mechanism.

Another noteworthy development is from Alibaba with Qwen models [5]. The release of Qwen2.5 marks a key intermediate step in Alibaba's evolution of reasoning ability models. It first provides a solid foundation for the model's common sense, expert knowledge, and reasoning abilities by expanding the high-quality pre-trained dataset from 7 trillion tokens to 18 trillion tokens. Then, Qwen2.5 introduces explicitly reasoning-oriented instruction fine-tuning data, which is typically presented through carefully crafted and structured prompts designed to guide the model to explicitly generate reasoning steps. Qwen2.5's training data contains a large number of mathematical, logical, and structured reasoning tasks, supplemented by detailed solutions and multi-step reasoning paths. The Qwen2.5 family of models has achieved competitive scores in several benchmarks such as GPQA and MMLU. 

Building on Qwen 2.5, Alibaba released Qwen3 in 2025, integrating a number of significant advances that take reasoning to a higher level. One of the major improvements in Qwen3 is the integration of “Thinking Mode”. Unlike Qwen 2.5, which relied heavily on explicit prompts, Qwen3 embeds special token separators &lt;think&gt; and &lt;/think&gt; directly in the training prompts. This mechanism allows the model to make a clear internal distinction between standard response generation and a deliberate step-by-step reasoning process. Thus, during training, the model learns to recognize when explicit reasoning is required. 

The Post-training of Qwen3 consists of a four-stage process and a strong-to-weak distillation strategy. For the four-stage process, the first stage is a long-chain of thought cold start. The dataset is constructed by carefully screening complex multi-domain problems in math, code, etc., and filtering the queries with Qwen2.5-72B-Instruct. The responses are then generated and filtered by QwQ-32B, and the model is initially trained with a small number of samples and steps to inject basic reasoning capabilities. The second stage is reasoning reinforcement learning. It selects difficult query-verifier pairs not used for cold start and applies the GRPO algorithm [12] to update the model parameters. The third stage is thinking mode fusion, which integrates “non-thinking modes” with “thinking modes” to construct a dataset containing both modes. Chat templates were then designed, introducing /think and /no_think symbols to enable mode switching, and also generating a thinking budget control capability. The fourth stage is generalized reinforcement learning, which utilizes a reward system covering more than 20 tasks to improve the model's ability to follow instructions, format adaptation, and preference alignment in multiple scenarios. 

Strong-to-weak distillation strategies are used to optimize the lightweight model [13]. In the offline distillation stage, response distillation is performed by combining the outputs of the teacher model in thinking and non-thinking modes, which helps the lightweight student model to master the basic reasoning and mode switching abilities. Subsequently, in the online distillation stage, the student model generates responses in thinking or non-thinking mode, and the student model is fine-tuned by aligning the output of the teacher model to minimize the KL dispersion. 

As we can see, there is a trade-off. Pure prompting approaches (e.g., CoT) are easy to deploy but may require very large models and skillful prompting to work well, while integrated reasoning models offer better performance and ease of use, but at the cost of more complex training and often more resources. Next, we will present a practical application of these reasoning strategies to LLMs through an experiment: we will use Qwen-3 8B to illustrate how forcing a model to think or not think affects its ability to solve problems, and how simple technologies such as truncated reasoning or multiple attempts at voting can make a difference. 

## Reasoning Modes in Qwen3-8B

In order to explore the effects of “reasoning modes” on LLMs, we conducted experiments using the Qwen3-8B reasoning model. Our motivation comes from the observation that although reasoning improves the accuracy of large language models, LLMs sometimes overthink simple questions [14]. If free reasoning is allowed, the model may generate unnecessarily complex chains of thought about a question whose actual answer is straightforward, and may make mistakes in the process. This phenomenon has been noted in previous work. For example, a recent survey on the efficiency of reasoning found that large models tend to “over-verify” even when they find the correct answer to a simple query, thus wasting computation and sometimes introducing errors [15]. In other words, more reasoning is not always better-especially for simple tasks, where short and straightforward approaches may be more efficient. We hope to quantify this and test some mitigation strategies. 

## Experimental Setup

Qwen3-8B is a model that supports thinking mode. In think mode, the model generates a chain of thoughts (labeled by &lt;think&gt; and &lt;/think&gt;) before giving the final answer. We set up the following different reasoning behaviors: 

Thinking Mode: The model runs in thinking mode without any restrictions. It can generate an arbitrarily long chain of thoughts before arriving at an answer. 

Truncated Thinking: Here, we still use the thinking mode, but impose a limit or truncation on the length of the chain of thoughts. In practice, this means that if the model does not arrive at an answer after a certain number of reasoning steps or tokens, we stop the reasoning process and force it to reach a conclusion. 

Truncated + Voting: This method combines the truncation method described above with self-consistent majority voting. We run the truncated thought process five times and then take a majority vote on the final answer that is generated. 

Truncated + Best-of-N: In this variant, we also run the model five times with truncated reasoning, but instead of a majority vote, we take the result of the maximum confidence (log probability) in five attempts. 

Prompt No-thinking: We follow Ma et al. [16] by replacing the reasoning chain with ‘Okay, I think I have finished thinking.’ and then let the model generate the final answer. 

## Results

We tested these methods on two well-known mathematical benchmarks:

GSM8K [17]: a dataset of 8000 elementary school math word problems. These problems are typically natural language arithmetic and logic problems of simple difficulty. 

MATH-500 [18]: A set of 500 problems drawn from the MATH dataset which is a collection of high school math competition problems. These tend to be quite challenging, often requiring creative multi-step solutions. 

The results are shown in Table 1. 

|  Qwen3-8B  | GSM8K Accuracy  | GSM8K Tokens Usage | MATH Accuracy | MATH Tokens Usage |
| --- | --- | --- | --- | --- |
| Thinking Mode  | 94.31  | 2.36M  | 94.60  | 2.64M  |
| Truncated <br>Thinking  | 76.12  | 0.67M  | 69.20  | 0.51M  |
| Truncated + <br>Voting  | 75.97  | 0.78M  | 67.40  | 0.55M  |
| Truncated + <br>Best-of-N  | 74.60  | 0.78M  | 69.20  | 0.55M  |
| Prompt <br>No-thinking  | 92.34  | 0.45M  | 84.80  | 0.60M  |

Table 1: Results on two representative mathematical benchmarks.

From the results, we have the following key observations:

**Thinking mode yields the highest accuracy** on both GSM8K (94.31%) and MATH-500 (94.60%)—but at a very high token cost (2.36M and 2.64M tokens). Full thinking mode allows Qwen3-8B to solve complex problems extremely well, but the long chains of thought are computationally expensive. 

**Truncated Thinking sharply degrades accuracy**, especially on MATH-500. On GSM8K, the accuracy drops to 76.12%, while on MATH-500 it drops further to 69.20%. This suggests that truncated reasoning is detrimental to the performance of both datasets, while MATH-500 is more affected because it contains inherently more difficult problems that require in-depth multi-step reasoning. 

**Voting and Best-of-N provide minimal gains when reasoning is truncated.** On both datasets, applying these methods after truncation failed to recover the accuracy loss. This suggests that if the chain of reasoning is hard truncated, the best answer attempted is often wrong. 

**Prompt No-thinking performs surprisingly well.** On GSM8K, it achieves 92.34% with the lowest token usage (0.45M), nearly matching full reasoning accuracy. On MATH-500, it still achieves 84.80%—which is much higher than any of the truncated reasoning variants. This may suggest that Qwen3-8B has internalized strong implicit reasoning ability. Even without explicitly generating chains-of-thought, it can answer many problems correctly, especially on GSM8K. For MATH-500, explicit reasoning still provides a notable advantage (Thinking 94.6% vs. No-thinking 84.8%), but the gap is not as large as one might expect. 

We also demonstrated an example on Table 2.

In summary, full reasoning is extremely powerful but token-inefficient. Truncated thinking can seriously hurt performance. So simply limiting the length of the chain of thought is not a feasible solution. Voting and Best-of-N methods offer little help once the chain is cut short. Prompt No-thinking is highly competitive suggesting that well-trained models like Qwen3-8B can already perform much of their reasoning implicitly. Based on the results, there is a significant opportunity for adaptive reasoning control—models should learn when to reason deeply and when to answer directly, instead of using reasoning mode indiscriminately. 

| Methods  | Question: Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to <br>download the file?  |
| --- | --- |
| Thinking Mode  | &lt;think&gt;\nOkay, let's try to figure out how long it takes Carla to download the 200 GB file. Hmm, so normally she downloads at 2 GB per minute. But there's a catch: 40% of the way through the download, her computer restarts, which takes 20 minutes. And then she has to restart the download from the beginning. So I need to calculate the total time including the restart time and the download time.\n\nFirst, let me break down the problem. <br>…… <br>Let me check with another way.\n\nLet me think of it as two phases: the first phase is downloading 80 GB, then the restart, then the second phase is downloading the entire file again. So total time is 40 + 20 + 100 = 160 minutes.\n\nYes, I think that's the correct approach. So the final answer is 160 minutes.\n&lt;/think&gt;\n\nCarla is downloading a 200 GB file at a rate of 2 GB per minute. The process is interrupted at 40% of the download, which is 80 GB, requiring a 20-minute restart. After the restart, she must restart the download from the beginning, meaning she has to download the entire file again.\n\n1. First Download Phase: \n Carla downloads 80 GB at 2 GB/minute: \n &#36;&#36;\n \\frac{80 \\text{ GB}}{2 \\text{ GB/minute}} = 40 \\text{ minutes}\n &#36;&#36;\n\n2. Restart Time: \n The restart takes 20 minutes.\n\n3. Second Download Phase: \n After the restart, Carla restarts the download from the beginning, requiring her to download the full 200 GB again: \n &#36;&#36;\n \\frac{200 \\text{ GB}}{2 \\text{ GB/minute}} = 100 \\text{ minutes}\n &#36;&#36;\n\n4. Total Time: \n Adding all the phases together: \n &#36;&#36;\n 40 \\text{ minutes} + 20 \\text{ <minutes} + 100 \\text{ minutes} = 160 \\text{ minutes}\n &#36;&#36;\n\nFinal Answer: 160 minutes (Correct)  |
| Truncated Thinking  | &lt;think&gt;\nOkay, let's try to figure out how long it takes Carla to download the 200 GB file. Hmm, so normally she downloads at 2 GB per minute. But there's a problem: 40% of the way through the download, Windows forces a restart, which takes 20 minutes. Then she has to restart the download from the beginning. So I need to calculate the total time taken considering this restart.\n\nFirst, let me break down the problem. <br>…… <br>Wait, maybe I need to clarify.\n\nWait, the problem says \"Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning.\" So, when the restart happens, she loses all progress? Or does she have to restart the download\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n&lt;/think&gt;.\n\nFinal Answer: 140 minutes\n\nStep-by-Step Explanation:\n\n1. Total File Size: 200 GB \n2. Normal Download Speed: 2 GB/minute \n3. Time to Download 40% (80 GB): \n &#36; \\frac{80 \\text{ GB}}{2 \\text{ GB/minute}} = 40 \\text{ minutes} &#36; \n4. Restart Duration: 20 minutes \n5. After Restart, Carla Must Redownload Entire File: \n &#36; \\frac{200 \\text{ GB}}{2 \\text{ GB/minute}} = 100 \\text{ minutes} &#36; \n6. Total Time: \n &#36; 40 \\text{ minutes} + 20 \\text{ minutes} + 100 \\text{ minutes} = 140 \\text{ minutes} &#36; \n\nFinal Answer: 140 minutes (Wrong)  |
| Truncated + Voting  | &lt;think&gt;\nOkay, let's see. Carla is downloading a 200 GB file. Normally, she can download at 2 GB per minute. But there's a catch: 40% of the way through the download, Windows forces a restart, which takes 20 minutes. Then she has to restart the download from the beginning. <br>…… <br>If the restart only pauses the download, she would resume from 80 GB. But the problem says \"has to restart the download from the beginning,\" which could mean she has to start over, leading to a full redownload. \n\nFinal Answer: 120 minutes. \n\nFinal Answer: 120 minutes. (Wrong)  |
| Truncated + Best-of-N  | &lt;think&gt;\nOkay, let me try to figure out how long it takes Carla to download the 200 GB file. Hmm, so normally she downloads at 2 GB per minute. But there's a catch: 40% of the way through the download, her computer restarts, which takes 20 minutes, and then she has to start over. <br>…… <br>Then she has to restart the download from the beginning. So that means after the restart, she starts over, so she has to download the entire 200 GB again.\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n&lt;/think&gt;.\n\nFinal Answer: 120 minutes\n\nStep-by-step explanation:\n\n1. Total file size: 200 GB \n2. Normal download speed: 2 GB/minute \n3. Time to download 40% of the file (80 GB): \n &#36; \\frac{80} (Wrong)  |
| Prompt No-thinking  | &lt;think&gt;\nOkay, I think I have finished thinking.\n&lt;/think&gt;\n\nTo determine how long it takes Carla to download the entire 200 GB file, let's break the problem down step by step.\n\n---\n\n### Step 1: Determine the point where the restart happens\n\nCarla is downloading a 200 GB file. <br>…… <br> Step 5: Total time**\n\nNow, we add up all $20$ minutes $\ln -$ Time the time:\n\n- Time to download 80 GB: $40$ minutes\n- Time for restart: 20 minutes\n- Time to download 200 GB again: 100 minutes\n\n$$\n\\text{Total time} = 40 + 20 + 100 = 160 \\text{ minutes}\n$$\n\n---\n\n### Final Answer:\nFinal Answer: 160 minutes (Correct)|

Table 2: Generation of different methods on a simple question from the GSM8K benchmark. Only thinking mode and prompt no-thinking methods answer the question correctly. 

## Conclusion

The evolution of LLM reasoning over the last few years has been remarkable. We have moved from models that can barely handle two-digit addition to models that can achieve Olympic-level results in math competitions. The evolution of LLM from simple thought chain prompts to complex “reasoning optimization” suggests that we are beginning to capture a form of machine logic reasoning in the black box of these neural networks. Our report covered how prompting methodologies like CoT, scratchpads, self-consistency, and least-to-most paved the way by revealing latent capabilities. The latest models like o1, Gemini 2.5, and Qwen-3 represent a new breed of LLMs that treat reasoning as a native ability, with features to automatically manage the reasoning process within a single model. 

Then, our experiments reveal important insights about the current state of reasoning in large language models such as Qwen3-8B. The experimental results show that reasoning in LLMs has reached an impressive level, but effective reasoning control remains an open challenge. Models like Qwen3-8B demonstrate that reasoning is often implicit and overthinking can lead to unnecessary compute cost. While cutting off reasoning hurts performance more than it saves tokens. We believe that dynamic reasoning control is crucial. Rather than always calling for complete reasoning or applying a fixed-length budget, future models should dynamically adjust the degree of reasoning based on task difficulty and internal confidence. 

## References

[1] Feng, S., Fang, G., Ma, X., & Wang, X. (2025). Efficient reasoning models: A survey. arXiv preprint arXiv:2504.10903. 

[2] Li, Z. Z., Zhang, D., Zhang, M. L., Zhang, J., Liu, Z., Yao, Y., ... & Liu, C. L. (2025). From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419. 

[3] Ahn, J., Verma, R., Lou, R., Liu, D., Zhang, R., & Yin, W. (2024). Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157. 

[4] Kumar, K., Ashraf, T., Thawakar, O., Anwer, R. M., Cholakkal, H., Shah, M., ... & Khan, S.(2025). Llm post-training: A deep dive into reasoning large language models. arXiv preprint arXiv:2502.21321. 

[5] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., ... & Qiu, Z. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388. 

[6] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35, 24824-24837. 

[7] Lanchantin, J., Toshniwal, S., Weston, J., & Sukhbaatar, S. (2023). Learning to reason and memorize with self-notes. Advances in Neural Information Processing Systems, 36, 11891-11911. 

[8] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2022). Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171. 

[9] Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., ... & Chi, E. (2022). Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625. 

[10] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., ... & McGrew, B.(2023). Gpt-4 technical report. arXiv preprint arXiv:2303.08774. 

[11] Team, G., Anil, R., Borgeaud, S., Alayrac, J. B., Yu, J., Soricut, R., ... & Blanco, L. (2023).Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. 

[12] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., ... & Guo, D. (2024). Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. 

[13] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., ... & He, Y. (2025). Deepseek-r1:Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. 

[14] Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., ... & Yu, D. (2024). Do not think that much for $2+3=?$ on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187. 

[15] Wang, R., Wang, H., Xue, B., Pang, J., Liu, S., Chen, Y., ... & Wong, K. F. (2025). Harnessing the reasoning economy: A survey of efficient reasoning for large language models.arXiv preprint arXiv:2503.24377. 

[16] Ma, W., He, J., Snell, C., Griggs, T., Min, S., & Zaharia, M. (2025). Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858. 

[17] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., ... & Schulman, J.(2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. 

[18] Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., ... & Cobbe, K. (2023, May). Let's verify step by step. In The Twelfth International Conference on Learning Representations. 

---