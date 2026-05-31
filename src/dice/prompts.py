"""Prompt templates for knowledge extraction and ReAct inference."""

from __future__ import annotations


REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


QUESTION_TYPE_TEMPLATE = """Summarize the type of the question. Here are some examples. Please only give me the part after "Type:" and make sure it's all in one paragraph without any line break:

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Type: Get a feature of some entity.

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Type: Answer which option has a feature.

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Type: Compare features of two entities.

Question: Which magazine was started first Arthur's Magazine or First for Women?
Type: Answer which option has a feature.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Type: Compare features of two entities.

Question: Alfie Allen played Theon Greyjoy on which show?
Type: Get a feature of some entity.

Question: """


QUESTION_KNOWLEDGE_TEMPLATE = """Please help me to summarize the required knowledge of the question. Here are some examples. Please only give me the part after "Knowledge:" and make sure it's all in one paragraph without any line break:

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Knowledge: The question requests a feature of an entity, which has another feature. To solve this question, we need to first find out what the entity is, by searching the entity's feature and look up the entity we want. Then, after obtaining the entity, we search it and find out the feature we need.

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Knowledge: The question requests a feature of an entity. To solve this question, we need to first search the entity, then lookup the feature we want, and find out the feature we need.

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Knowledge: The question asks us to make a choice between two options, which possess a certain feature. To solve this question, we need to first search the first entity, then search the second entity, and figure out which one possess the feature we are looking for. When searching the first entity, it could not be found, so we search another entity that is closely related to our entity. By searching this entity, we find it does not possess the feature we are looking for, so we choose the other entity as the answer for the question.

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Knowledge: The question asks what feature two entities have in common. To solve this question, we first search the first entity and get the feature we need, then search the second entity and get the feature we need. By comparing their feature set, we find out the answer of the question.

Question: Which magazine was started first Arthur's Magazine or First for Women?
Knowledge: The question asks us to compare a feature of two entities. To solve this question, we first search the first entity and get the feature we need, then search the second entity and get the feature we need. By comparing their feature set, we find out the answer of the question.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Knowledge: The question asks us to compare a feature of two entities. To solve this question, we first search the first entity and get the feature we need, then search the second entity and get the feature we need. By comparing their feature set, we find out the answer of the question.

Question: """


def clean_generation(text: str, prefix: str) -> str:
    """Remove optional labels/markdown from an extraction response."""

    text = " ".join(text.strip().split())
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    if "**" in text:
        text = text.split("**", 1)[0].strip()
    return text


def clean_question_type(text: str) -> str:
    return clean_generation(text, "Type:")


def clean_question_knowledge(text: str) -> str:
    return clean_generation(text, "Knowledge:")

