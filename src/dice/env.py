"""Wikipedia environment used by ReAct/DICE experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


def clean_str(text: str) -> str:
    """Normalize escaped text returned by Wikipedia pages."""

    return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


@dataclass
class StepResult:
    observation: str
    reward: int
    done: bool
    info: dict


class WikiEnv:
    """A small text environment with search, lookup, think, and finish actions."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.page: str | None = None
        self.obs: str | None = None
        self.lookup_keyword: str | None = None
        self.lookup_list: list[str] | None = None
        self.lookup_cnt: int | None = None
        self.steps = 0
        self.answer: str | None = None
        self.search_time = 0.0
        self.num_searches = 0
        self.result_titles: list[str] = []
        self.reset()

    def reset(self) -> str:
        self.obs = "Interact with Wikipedia using search[], lookup[], and finish[].\n"
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        return self.obs

    def _get_info(self) -> dict:
        return {"steps": self.steps, "answer": self.answer}

    def construct_lookup_list(self, keyword: str) -> list[str]:
        if self.page is None:
            return []

        paragraphs = [p.strip() for p in self.page.split("\n") if p.strip()]
        sentences: list[str] = []
        for paragraph in paragraphs:
            sentences.extend(paragraph.split(". "))
        sentences = [sentence.strip() + "." for sentence in sentences if sentence.strip()]
        return [sentence for sentence in sentences if keyword.lower() in sentence.lower()]

    @staticmethod
    def get_page_obs(page: str) -> str:
        paragraphs = [p.strip() for p in page.split("\n") if p.strip()]
        sentences: list[str] = []
        for paragraph in paragraphs:
            sentences.extend(paragraph.split(". "))
        sentences = [sentence.strip() + "." for sentence in sentences if sentence.strip()]
        return " ".join(sentences[:5])

    def search_step(self, entity: str) -> None:
        entity_query = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_query}"

        start = time.time()
        response = requests.get(search_url, timeout=self.timeout)
        self.search_time += time.time() - start
        self.num_searches += 1
        response.raise_for_status()

        soup = BeautifulSoup(response.text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            return

        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):
            self.search_step(f"[{entity}]")
            return

        self.page = ""
        for paragraph in page:
            if len(paragraph.split()) > 2:
                self.page += clean_str(paragraph)
                if not paragraph.endswith("\n"):
                    self.page += "\n"

        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None

    def step(self, action: str) -> StepResult:
        reward = 0
        done = False
        action = action.strip()

        if self.answer is not None:
            return StepResult(self.obs or "", reward, True, self._get_info())

        if action.startswith("search[") and action.endswith("]"):
            self.search_step(action[len("search[") : -1])
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup[") : -1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0

            assert self.lookup_list is not None
            assert self.lookup_cnt is not None
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = (
                    f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                    + self.lookup_list[self.lookup_cnt]
                )
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            self.answer = action[len("finish[") : -1]
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = f"Invalid action: {action}"

        self.steps += 1
        return StepResult(self.obs or "", reward, done, self._get_info())

    def get_time_info(self) -> dict:
        return {
            "call_speed": self.search_time / self.num_searches if self.num_searches else 0,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }

