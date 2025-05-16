# JIR-Arena

> Code for paper [JIR-Arena: The First Benchmark Dataset for Just-in-time Information Recommendation]().

Just-in-time Information Recommendation (JIR) is a service that delivers the most relevant information precisely when users need it the most. It plays a critical role in filling users’ information gaps during pivotal moments like those in learning, work, and social interactions, thereby enhancing decision-making quality and life efficiency with minimal user effort. 

Recent device-efficient deployment of performant foundation models and the proliferation of intelligent wearable devices have made the realization of always-on JIR assistants feasible. However, despite the potential of JIR systems to transform our daily life, there has been no prior systematic effort to formally define JIR tasks or establish evaluation frameworks. 

To bridge this gap, we present the first comprehensive mathematical definition of JIR tasks and their associated evaluation metrics. Furthermore, we introduce JIR-Arena, the first multimodal JIR benchmark dataset with diverse and information-request-intensive scenarios, designed to evaluate JIR systems across multiple dimensions, including whether they can i) accurately infer user information needs, ii) provide timely and helpfully relevant recommendations, and iii) effectively avoid the inclusion of irrelevant content that might distract users. 

Additionally, we implement a baseline JIR system that processes multimodal sensory information streams consistent with user inputs and provides real-time JIR instances. Our evaluation of the baseline on JIR-Arena reveals that while large foundation model-based JIR systems can simulate user needs with reasonable recall, they struggle with precision and effective content retrieval.

In summary, this code repo includes the implementation of:

* The user information need simulation with AI models.
* The JIR instance completion through information retrieval.
* The baseline JIR systems.
* The evaluation metrics.

Additionally, it includes crucial data including:
* The metainfo of scenes in JIR-Arena.
* Video trancripts and narratives.
* The benchmark ground truth.
* Baseline runs.

You can find our knowledge bases for the information retrieval stage [here](https://huggingface.co/datasets/EmpathYang/JIR-Arena).