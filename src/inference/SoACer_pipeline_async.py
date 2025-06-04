import asyncio
import aiohttp
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import logging

from boilerpy3 import extractors
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

import torch
import torch.nn as nn
import numpy as np
import json
import os
import random

import logging
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from boilerpy3 import extractors
import argparse
import nest_asyncio

class AsyncSoACerCrawler:
    """
    Robust asynchronous web crawler with:
      - Per-domain aiohttp session
      - Resilient to TLS teardown / anti-bot protection
      - Retry logic, optional link BFS
    """
    def __init__(self, user_agent=None, max_links=3, concurrency=3, retries=2, crawl_delay=0.3):
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
        self.max_links = max_links
        self.concurrency = concurrency
        self.retries = retries
        self.crawl_delay = crawl_delay
        self._robots_cache = {}

    async def _fetch_html(self, url: str, session: aiohttp.ClientSession) -> (str, str):
        """
        Attempt to fetch HTML with retry and exponential backoff.
        Returns (html, error_message).
        """
        for attempt in range(self.retries + 1):
            try:
                async with session.get(url, timeout=60) as resp:
                    if resp.status != 200:
                        return None, f"HTTP error {resp.status} on {url}"
                    return await resp.text(), None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt * 0.5)
                    continue
                return None, f"Request error on {url}: {str(e)}"
            except Exception as e:
                return None, f"Unexpected error on {url}: {str(e)}"

    async def _allowed_by_robots(self, url: str, session: aiohttp.ClientSession) -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_cache.get(domain)

        if rp is None:
            rp = RobotFileParser()
            robots_url = f"{domain}/robots.txt"
            try:
                async with session.get(robots_url, timeout=60) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        rp.parse(text.splitlines())
                    else:
                        rp.allow_all = True
            except Exception:
                rp.allow_all = True
            self._robots_cache[domain] = rp

        can_fetch = rp.can_fetch(self.user_agent, url)
        if not can_fetch:
            logging.warning(f"robots.txt disallows fetching: {url}")
        return True  # Always try, just log warning

    def _extract_content(self, html: str, url: str) -> str:
        try:
            extractor = extractors.ArticleExtractor()
            return extractor.get_doc(html).content
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
            return "\n".join(lines)

    async def _crawl_one(self, url: str, session: aiohttp.ClientSession) -> (str, list, str):
        await self._allowed_by_robots(url, session)
        html, error = await self._fetch_html(url, session)
        if error:
            return None, [], error

        cleaned = self._extract_content(html, url)

        # Skip BFS if disabled
        if self.max_links == 0:
            return cleaned, [], None

        soup = BeautifulSoup(html, "html.parser")
        domain_netloc = urlparse(url).netloc
        hrefs = []

        for tag in soup.find_all("a", href=True):
            href = urljoin(url, tag.get("href"))
            if urlparse(href).netloc != domain_netloc:
                continue
            if any(k in href.lower() for k in ['privacy', 'terms', 'policy']):
                continue
            hrefs.append(href)

        prioritized = [h for h in hrefs if any(p in h.lower() for p in ['about', 'service'])]
        remaining = [h for h in hrefs if h not in prioritized]
        child_links = prioritized[:self.max_links] + remaining[:self.max_links]

        return cleaned, child_links, None

    async def crawl(self, seed_url: str) -> str:
        """
        Perform a 2-level BFS starting from the seed URL.
        Returns combined text or None.
        """
        if not seed_url.startswith("http"):
            seed_url = "http://" + seed_url

        visited = set()
        texts = {}
        queue = [seed_url]

        async with aiohttp.ClientSession(headers={"User-Agent": self.user_agent}) as session:
            for depth in range(4):
                tasks = []
                for link in queue:
                    if link in visited:
                        continue
                    visited.add(link)
                    tasks.append(self._crawl_one(link, session))

                if not tasks:
                    break

                results = await asyncio.gather(*tasks, return_exceptions=True)
                next_queue = []

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.warning(f"Exception crawling {queue[i]}: {str(result)}")
                        continue

                    cleaned, child_links, error = result
                    if cleaned:
                        texts[queue[i]] = cleaned
                    elif error:
                        logging.warning(error)

                    next_queue.extend(child for child in child_links if child not in visited)

                queue = next_queue
                await asyncio.sleep(self.crawl_delay)

        combined = "\n".join(texts.values())
        return combined if combined.strip() else None



class DownstreamModelSingle(nn.Module):
    """
    The same classifier architecture used in training.
    """

    def __init__(self, embed_size: int, class_num: int, common_dim: int = None):
        super().__init__()
        if common_dim and (common_dim != embed_size):
            self.compress = nn.Sequential(
                nn.Linear(embed_size, common_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            final_input_dim = common_dim
        else:
            self.compress = nn.Identity()
            final_input_dim = embed_size

        self.fc1 = nn.Linear(final_input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, class_num)

    def forward(self, x):
        out = self.compress(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        return self.fc3(out)


class SoACerPredictor:
    """
    Predictor that wraps:
      - AsyncSoACerCrawler for async crawling
      - Summarization via Sumy (blocking, offloaded to executor)
      - Embedding + classification via HuggingFace (blocking, offloaded)
    """

    def __init__(
        self,
        embedder_name="meta-llama/Meta-Llama-3-8B",
        classifier_source="Shahriar/SoACer",
        classifier_local_dir=None,
        device=None,
        crawler_max_links=3,
        crawler_concurrency=10,
    ):
        # PyTorch device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Label names (must match the classifier training)
        self.label_names = [
            "finance, marketing & human resources",
            "information technology & electronics",
            "consumer & supply chain",
            "civil, mechanical & electrical",
            "medical",
            "sports, media & entertainment",
            "education",
            "government, defense & legal",
            "travel, food & hospitality",
            "non-profit",
        ]

        # Async crawler
        self.async_crawler = AsyncSoACerCrawler(
            user_agent=None,  # uses realistic browser UA by default
            max_links=crawler_max_links,
            concurrency=crawler_concurrency,
        )

        # Load embedder + tokenizer
        self.embedder_name = embedder_name
        self.classifier_source = classifier_source
        self.classifier_local_dir = classifier_local_dir

        self.embedding_model, self.tokenizer = self._load_embedder()
        self.classifier = self._load_classifier()
        self.classifier.eval()

        # Use the existing event loop
        self.loop = asyncio.get_event_loop()

    def _load_embedder(self):
        """
        Load tokenizer and causal LM for embeddings. Returns (model, tokenizer).
        """
        tokenizer = AutoTokenizer.from_pretrained(self.embedder_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        config_kwargs = {
            "trust_remote_code": True,
            "revision": "main",
            "use_auth_token": None,
            "output_hidden_states": True,
        }
        model_config = AutoConfig.from_pretrained(self.embedder_name, **config_kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            self.embedder_name,
            config=model_config,
            device_map=None,            # No automatic sharding
            torch_dtype=torch.float16,  # Use float16 for efficiency
            attn_implementation="eager",
        )
        model.to(self.device)
        return model, tokenizer

    def _load_classifier(self):
        """
        Load the downstream classifier from either a local directory or HuggingFace Hub.
        """
        if self.classifier_local_dir:
            config_path = os.path.join(self.classifier_local_dir, "config.json")
            model_path = os.path.join(self.classifier_local_dir, "pytorch_model.bin")
        else:
            config_path = hf_hub_download(repo_id=self.classifier_source, filename="config.json")
            model_path = hf_hub_download(repo_id=self.classifier_source, filename="pytorch_model.bin")

        with open(config_path) as f:
            config = json.load(f)

        model = DownstreamModelSingle(
            embed_size=config["embed_size"],
            class_num=config["class_num"],
            common_dim=config.get("common_dim", config["embed_size"]),
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model = model.half()  # Convert weights to float16
        return model

    def summarize_text(self, text: str, sentences_count=12) -> str:
        """
        Synchronous summarization with Sumy (LexRank).
        If fewer than 50 chars, returns "NaN".
        """
        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        summarizer = LexRankSummarizer()
        summarizer.threshold = 0.1
        summarizer.epsilon = 0.05
        summary = summarizer(parser.document, sentences_count)
        summarized_text = " ".join(str(sentence) for sentence in summary)
        return summarized_text if len(summarized_text) >= 50 else "NaN"

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Synchronous embedding: tokenize, run through LM, and mean-pool last hidden state.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return torch.mean(outputs.hidden_states[-1], dim=1)

    def _classify_from_embedding(self, embedding: torch.Tensor, url: str, final_text: str) -> dict:
        """
        Synchronous classification from an embedding. Returns the result dict.
        """
        with torch.no_grad():
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_label_index = int(np.argmax(probs))
        top_label = self.label_names[top_label_index]
        return {
            "predicted_label": top_label,
            "all_probabilities": dict(zip(self.label_names, probs.tolist())),
            "summaries": final_text,
            "url": url,
        }

    async def __call_async__(self, input_str: str) -> dict:
        """
        Asynchronous pipeline:
          1. Crawl if input_str is a URL
          2. Summarize (offloaded to executor)
          3. Embed text (offloaded)
          4. Classify embedding (offloaded)
        Returns a dict with predicted_label, all_probabilities, summaries, url or an error.
        """
        # Detect if input_str is a URL
        def is_url(text: str) -> bool:
            parsed = urlparse(text)
            return parsed.scheme in ("http", "https")

        def get_domain(input_url: str) -> str:
            parsed = urlparse(input_url)
            return f"{parsed.scheme}://{parsed.netloc}"

        # 1. Crawl if it's a URL
        if is_url(input_str):
            domain_url = get_domain(input_str)
            print(f"Starting crawl for: {domain_url}")
            try:
                combined_text = await self.async_crawler.crawl(domain_url)
            except Exception as e:
                return {"error": f"Unhandled scraping failure: {str(e)}", "url": input_str}

            if not combined_text or len(combined_text.strip()) < 50:
                return {
                    "error": "Unable to extract meaningful content from URL",
                    "url": input_str
                }
        else:
            combined_text = input_str

        # 2. Summarize (offload to executor)
        summarized = await self.loop.run_in_executor(None, self.summarize_text, combined_text)
        # If summarization was too short, use the raw combined_text instead
        final_text = summarized if summarized != "NaN" else combined_text

        # 3. Embed (offload to executor)
        try:
            embedding = await self.loop.run_in_executor(None, self.embed_text, final_text)
        except Exception as e:
            return {"error": f"Embedding failure: {str(e)}", "url": input_str}

        # 4. Classify (offload to executor)
        try:
            result = await self.loop.run_in_executor(
                None, self._classify_from_embedding, embedding, input_str, final_text
            )
            return result
        except Exception as e:
            return {"error": f"Classification failure: {str(e)}", "url": input_str}

    def __call__(self, input_str: str) -> dict:
        """
        Synchronous wrapper that runs the async pipeline to completion.
        """
        return self.loop.run_until_complete(self.__call_async__(input_str))


# Example: Classify 10 URLs from the Hugging Face dataset
async def classify_multiple_websites(url_list):
    predictor = SoACerPredictor(
        embedder_name="meta-llama/Meta-Llama-3-8B",
        classifier_source="Shahriar/SoACer",
        classifier_local_dir=None,
        device=None,
        crawler_max_links=3,
        crawler_concurrency=10,
    )
    tasks = [asyncio.create_task(predictor.__call_async__(url)) for url in url_list]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


nest_asyncio.apply()

def safe_filename(url: str) -> str:
    return (
        url.replace("http://", "")
        .replace("https://", "")
        .replace("/", "_")
        .replace(":", "_")
        .strip()
    )

async def classify_batch(urls, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    predictor = SoACerPredictor()

    for url in urls:
        url = url.strip()
        if not url:
            continue

        try:
            result = await predictor.__call_async__(url)
            rand_id = random.randint(100000, 999999)
            output_path = os.path.join(output_dir, f"result_{rand_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[✓] Saved: {output_path}")
        except Exception as e:
            print(f"[!] Failed to classify {url}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Batch classify URLs using SoACerPredictor.")
    parser.add_argument("--input_txt", type=str, required=True, help="Path to .txt file with one URL per line.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save JSON output.")

    args = parser.parse_args()

    if not os.path.exists(args.input_txt):
        raise FileNotFoundError(f"Input file not found: {args.input_txt}")

    with open(args.input_txt, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    loop = asyncio.get_event_loop()
    loop.run_until_complete(classify_batch(urls, args.output_dir))


if __name__ == "__main__":
    main()
