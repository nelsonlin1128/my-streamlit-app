from icrawler.builtin import GoogleImageCrawler
import os

output_dir = "data/styles/jojo"
os.makedirs(output_dir, exist_ok=True)

crawler = GoogleImageCrawler(storage={'root_dir': output_dir})
crawler.crawl(keyword="JoJo's Bizarre Adventure anime screenshot",
              max_num=1000,
              min_size=(200,200),
              max_size=None)

