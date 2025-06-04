from transformers import AutoTokenizer, AutoModel
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from huggingface_hub import hf_hub_download
from collections import defaultdict
import torch.nn as nn
import torch
import json
import numpy as np
import os
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoTokenizer,
    AutoModelForCausalLM
)

import requests
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import logging
import argparse


class SoACerCrawler:
    def __init__(self, user_agent="SoACerBot"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def is_allowed_by_robots(self, url):
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(self.user_agent, url)
        except Exception as e:
            logging.warning(f"Failed to parse robots.txt for {url}: {e}")
            return False

    def extract_content(self, html, url):
        try:
            from boilerpy3 import extractors
            extractor = extractors.ArticleExtractor()
            doc = extractor.get_doc(html)
            return doc.content
        except Exception:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            return '\n'.join(chunk.strip() for chunk in soup.get_text().splitlines() if chunk.strip())

    def fetch_and_clean(self, url):
        try:
            if not self.is_allowed_by_robots(url):
                return None, f"Disallowed by robots.txt: {url}"

            resp = self.session.get(url, timeout=10)
            if resp.status_code != 200:
                return None, f"HTTP error {resp.status_code} on {url}"

            html = resp.text
            cleaned = self.extract_content(html, url)
            return cleaned, None
        except requests.exceptions.RequestException as e:
            return None, f"Request error on {url}: {str(e)}"

    def crawl(self, url, max_links=3):
        if not url.startswith("http"):
            url = "http://" + url

        visited = set()
        texts = {}
        queue = [url]

        for depth in range(4):  # 2-level BFS
            next_queue = []
            for link in queue:
                if link in visited:
                    continue
                visited.add(link)
                text, error = self.fetch_and_clean(link)
                if text:
                    texts[link] = text
                elif error:
                    logging.warning(error)

                try:
                    html = self.session.get(link, timeout=10).text
                    soup = BeautifulSoup(html, "html.parser")
                    hrefs = [urljoin(link, tag.get("href")) for tag in soup.find_all("a", href=True)]
                    hrefs = [h for h in hrefs if urlparse(h).netloc == urlparse(url).netloc]
                    hrefs = [h for h in hrefs if all(k not in h.lower() for k in ['privacy', 'terms', 'policy'])]
                    prioritized = [h for h in hrefs if any(p in h.lower() for p in ['about', 'service'])]
                    remaining = [h for h in hrefs if h not in prioritized]
                    next_queue.extend(prioritized[:max_links] + remaining[:max_links])
                except Exception as e:
                    logging.warning(f"Failed to extract links from {link}: {e}")
            queue = next_queue

        combined = "\n".join(texts.values())
        return combined if combined.strip() else None


# DownstreamModelSingle remains the same as in training
class DownstreamModelSingle(nn.Module):
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
    def __init__(
        self,
        embedder_name="meta-llama/Meta-Llama-3-8B",
        classifier_source="Shahriar/SoACer",
        classifier_local_dir=None,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder_name = embedder_name
        self.classifier_source = classifier_source
        self.classifier_local_dir = classifier_local_dir
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


        self.embedding_model, self.tokenizer = self.load_embedder()
        # embedding = embedding.to(self.classifier.device)

        self.classifier = self.load_classifier()
        self.classifier.eval()

    def load_embedder(self):
        tokenizer = AutoTokenizer.from_pretrained(self.embedder_name, trust_remote_code=True)
        # model = AutoModel.from_pretrained(self.embedder_name, trust_remote_code=True)
        # If model doesn't have a pad_token, assign the eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"


        # ----- Model Config + Loading -----
        config_kwargs = {
            "trust_remote_code": True,
            "revision": 'main',
            "use_auth_token": None,
            "output_hidden_states": True  # Enable hidden state outputs
        }

        model_config = AutoConfig.from_pretrained(self.embedder_name, **config_kwargs)


        model = AutoModelForCausalLM.from_pretrained(
        self.embedder_name,
        config=model_config,
        # device_map="auto",        # <--- Automatic sharding across available GPUs
        device_map=None,  # Do not shard

        # device_map = device,
        torch_dtype=torch.float16,  # Use half precision for efficiency (optional)
        attn_implementation="eager"
    )

        model.to(self.device)
        return model, tokenizer

    def load_classifier(self):
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
            common_dim=config.get("common_dim", config["embed_size"])
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model = model.half()  # Convert weights to float16

        return model

    def summarize_text(self, text: str, sentences_count=12) -> str:
        parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
        summarizer = LexRankSummarizer()
        summarizer.threshold = 0.1  # Set a custom threshold
        summarizer.epsilon = 0.05   # Set a custom epsilon
        summary = summarizer(parser.document, sentences_count)
        summarized_text = ' '.join(str(sentence) for sentence in summary)
        return summarized_text if len(summarized_text) >= 50 else "NaN"




    def embed_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

          
        return torch.mean(outputs.hidden_states[-1], dim=1)

    def predict(self, embedding: torch.Tensor, hash_key: str, url: str, summarized_text: str) -> dict:
        with torch.no_grad():
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_classes = (probs > 0.5).astype(int)

        result = {
            hash_key: {
                "predicted_label": list(np.array(self.label_names)[np.where(predicted_classes == 1)[0]]),
                "all_probabilities": dict(zip(self.label_names, probs.tolist())),
                "summaries": summarized_text,
                "url": url
            }
        }
        return result
    def __call__(self, input: str):
        from urllib.parse import urlparse

        def is_url(text):
            return urlparse(text).scheme in ["http", "https"]

        def get_domain(input_url):
            parsed = urlparse(input_url)
            return f"{parsed.scheme}://{parsed.netloc}"

        if is_url(input):
            crawler = SoACerCrawler()
            domain_url = get_domain(input)

            try:
                combined_text = crawler.crawl(domain_url)
            except Exception as e:
                return {
                    "error": f"Unhandled scraping failure: {str(e)}",
                    "url": input
                }

            if not combined_text or len(combined_text.strip()) < 50:
                return {
                    "error": "Unable to extract meaningful content from URL",
                    "url": input
                }

            summarized_text = self.summarize_text(combined_text)
        else:
            summarized_text = self.summarize_text(input)

        if summarized_text == "NaN":
            return {
                "error": "Text too short to summarize",
                "url": input
            }

        try:
            embedding = self.embed_text(summarized_text)
            with torch.no_grad():
                logits = self.classifier(embedding)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_label_index = int(np.argmax(probs))
            top_label = self.label_names[top_label_index]

            return {
                "predicted_label": top_label,
                "all_probabilities": dict(zip(self.label_names, probs.tolist())),
                "summaries": summarized_text,
                "url": input
            }

        except Exception as e:
            return {
                "error": f"Classification failure: {str(e)}",
                "url": input
            }




def main():
    parser = argparse.ArgumentParser(description="Classify a URL or a text file using SoACerPredictor")
    parser.add_argument("--input", type=str, required=True, help="Either a URL or a path to a .txt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the result JSON")

    args = parser.parse_args()
    input_value = args.input
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load content from file if it's a text file
    if input_value.endswith(".txt") and os.path.exists(input_value):
        with open(input_value, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = input_value

    predictor = SoACerPredictor()
    result = predictor(input_text)

    # Save result
    safe_name = (
        input_value.replace("http://", "")
        .replace("https://", "")
        .replace("/", "_")
        .replace(":", "")
        .strip()
    )
    output_path = os.path.join(output_dir, f"{safe_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[✓] Prediction saved to: {output_path}")

if __name__ == "__main__":
    main()
