import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import config

class WebCrawler:
    def __init__(self, base_url, max_depth):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited = set()

    def save_page(self, url, content):
        parsed_url = urlparse(url)
        path = config.SCRAPED_DATA + parsed_url.path.strip("/").replace("/", "_")
        filename = f"{path}.html"
        if not filename:
            filename = "index.html"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)

    def crawl(self, url, depth):
        if depth > self.max_depth or url in self.visited:
            return
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return
        
        self.visited.add(url)
        self.save_page(url, response.text)

        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            next_url = urljoin(self.base_url, link['href'])
            if self.base_url in next_url:
                self.crawl(next_url, depth + 1)

def main():
    base_url = "https://docs.nvidia.com/cuda/"
    max_depth = 5
    crawler = WebCrawler(base_url, max_depth)
    crawler.crawl(base_url, 0)

if __name__ == "__main__":
    main()