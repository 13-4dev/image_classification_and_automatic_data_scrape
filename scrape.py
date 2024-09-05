import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib.request

def download_images(query, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_search_url = f"https://www.google.com/search?q={query}&tbm=isch&start="
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    urls = []

    for page in range(0, 5):
        search_url = base_search_url + str(page * 20)
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all("img")

        for img in img_tags:
            url = img.get("src")
            if url and url.startswith("http"):
                urls.append(url)

        if len(urls) >= 1000:
            break

    for i, url in enumerate(urls[:1000]):
        try:
            img_data = urllib.request.urlopen(url).read()
            with open(os.path.join(output_dir, f"{query}_{i+1}.jpg"), 'wb') as handler:
                handler.write(img_data)
            print(f"Downloaded: {query}_{i+1}.jpg")
        except Exception as e:
            print(f"Could not download {url}: {e}")
